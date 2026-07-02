"""
گام ۷ آموزش: تبدیل بلوک‌های ساختاریافته‌ی صفحات (خروجی page_processor) به
chunk های نهایی آماده برای embedding.

استراتژی (تصمیم‌گیری‌شده در گفتگوی آموزشی):
  1. بلوک‌ها درون هر فصل، بر اساس heading به "بخش" (section) گروه‌بندی می‌شوند.
     هر بخش = یک heading + تمام body/warning های زیر آن، تا قبل از heading بعدی.
  2. جدول‌ها همیشه همراه با heading (و مقدمه‌ی متنی) بخش خودشان می‌مانند و
     هرگز با بخش دیگری ادغام نمی‌شوند — چون جدول یک واحد معنایی یکپارچه است.
  3. بخش‌های متنی (غیر جدول) که کوتاه‌تر از MIN_CHUNK_CHARS هستند، با بخش
     بعدیِ همان فصل ادغام می‌شوند تا context کافی برای embedding داشته باشند.
     این ادغام هرگز از مرز یک فصل به فصل دیگر عبور نمی‌کند.
  4. هر chunk نهایی متادیتای کامل همراه خود دارد: شماره فصل، عنوان بخش،
     شماره صفحه (یا بازه‌ی صفحات)، نوع محتوا، و پرچم وابستگی به نمودار.
"""

import argparse
import json
from pathlib import Path

import fitz

from page_processor import process_page
from structure_extraction import assign_chapter_to_pages

MIN_CHUNK_CHARS = 150


def build_sections(doc: fitz.Document) -> list[dict]:
    """
    تمام صفحات سند را پردازش می‌کند و بلوک‌ها را به "بخش‌ها" (section)
    گروه‌بندی می‌کند: هر بخش با یک heading شروع می‌شود و تا heading بعدی
    (در همان فصل یا فصل بعدی) ادامه می‌یابد.

    خروجی: لیستی از دیکشنری‌های بخش، هر کدام شامل:
        chapter, heading, content_blocks (لیست بلوک‌های body/warning/table),
        start_page, end_page, has_diagram_reference
    """
    page_to_chapter = assign_chapter_to_pages(doc)
    sections = []
    current_section = None

    for page in doc:
        page_idx = page.number
        chapter = page_to_chapter.get(page_idx)
        result = process_page(page)
        page_number = result['page_number']
        has_diagram = result['has_diagram_reference']

        for block in result['blocks']:
            if block['type'] == 'heading':
                # شروع یک بخش جدید
                if current_section is not None:
                    sections.append(current_section)
                current_section = {
                    'chapter': chapter,
                    'heading': block['text'],
                    'content_blocks': [],
                    'start_page': page_number,
                    'end_page': page_number,
                    'has_diagram_reference': has_diagram,
                }
            else:
                # اگر هنوز هیچ heading ندیده‌ایم (مثل صفحات ابتدایی جلد)،
                # یک بخش بدون عنوان مشخص می‌سازیم تا محتوا گم نشود
                if current_section is None:
                    current_section = {
                        'chapter': chapter,
                        'heading': None,
                        'content_blocks': [],
                        'start_page': page_number,
                        'end_page': page_number,
                        'has_diagram_reference': has_diagram,
                    }
                current_section['content_blocks'].append(block)
                current_section['end_page'] = page_number
                current_section['has_diagram_reference'] = (
                    current_section['has_diagram_reference'] or has_diagram
                )

    if current_section is not None:
        sections.append(current_section)

    return sections


def section_to_text(section: dict) -> str:
    """متن قابل embedding یک بخش را می‌سازد: عنوان + محتوای آن."""
    parts = []
    if section['heading']:
        parts.append(section['heading'])
    for block in section['content_blocks']:
        if block['type'] == 'warning':
            parts.append(f"[{block['text']}]")
        else:
            parts.append(block['text'])
    return '\n'.join(parts).strip()


def merge_short_sections(sections: list[dict]) -> list[dict]:
    """
    بخش‌های کوتاه‌تر از MIN_CHUNK_CHARS را با بخش بعدیِ همان فصل ادغام می‌کند.
    بخش‌هایی که جدول دارند هرگز ادغام نمی‌شوند (طبق تصمیم طراحی).

    نکته (کشف‌شده در گام ۷ آموزش، بررسی کیفیت بعد از اولین اجرای کامل):
    اگر آخرین بخش یک فصل کوتاه باشد، نمی‌توان آن را با بخش بعدی ادغام کرد
    چون بخش بعدی متعلق به فصل دیگری است. در نسخه‌ی اول این تابع، چنین
    بخش‌هایی بدون ادغام در خروجی باقی می‌ماندند (مثل chunk فقط ۲۴ کاراکتری
    'کلیدهای / سیستم صوتی' در انتهای فصل ۲). راه‌حل: در این حالت خاص،
    بخش کوتاه را به‌جای ادغام «رو به جلو»، به آخرین بخش موجود در merged
    (که از همان فصل است) می‌چسبانیم — یعنی ادغام «رو به عقب».
    """
    merged = []
    pending = None

    for section in sections:
        if pending is not None:
            if pending['chapter'] == section['chapter']:
                # ادغام pending با بخش فعلی (ادغام رو به جلو)
                pending['content_blocks'].extend(
                    ([{'type': 'heading_inline', 'text': section['heading']}] if section['heading'] else [])
                    + section['content_blocks']
                )
                pending['end_page'] = section['end_page']
                pending['has_diagram_reference'] = (
                    pending['has_diagram_reference'] or section['has_diagram_reference']
                )
                section = pending
                pending = None
            else:
                # فصل عوض شده و نمی‌توان رو به جلو ادغام کرد.
                # اگر بخش هم‌فصلی قبلاً در merged ثبت شده، pending را
                # به انتهای آن می‌چسبانیم (ادغام رو به عقب).
                if merged and merged[-1]['chapter'] == pending['chapter']:
                    merged[-1]['content_blocks'].extend(
                        ([{'type': 'heading_inline', 'text': pending['heading']}] if pending['heading'] else [])
                        + pending['content_blocks']
                    )
                    merged[-1]['end_page'] = pending['end_page']
                    merged[-1]['has_diagram_reference'] = (
                        merged[-1]['has_diagram_reference'] or pending['has_diagram_reference']
                    )
                else:
                    # هیچ بخش هم‌فصلی برای ادغام پیدا نشد (مثلاً تنها بخش یک فصل کوتاه است)
                    merged.append(pending)
                pending = None

        has_table = any(b['type'] == 'table' for b in section['content_blocks'])
        text_len = len(section_to_text(section))

        if not has_table and text_len < MIN_CHUNK_CHARS:
            pending = section
        else:
            merged.append(section)

    if pending is not None:
        # همان منطق ادغام رو به عقب برای pending باقی‌مانده در پایان سند
        if merged and merged[-1]['chapter'] == pending['chapter']:
            merged[-1]['content_blocks'].extend(
                ([{'type': 'heading_inline', 'text': pending['heading']}] if pending['heading'] else [])
                + pending['content_blocks']
            )
            merged[-1]['end_page'] = pending['end_page']
            merged[-1]['has_diagram_reference'] = (
                merged[-1]['has_diagram_reference'] or pending['has_diagram_reference']
            )
        else:
            merged.append(pending)

    return merged


def sections_to_chunks(sections: list[dict]) -> list[dict]:
    """بخش‌های نهایی را به فرمت chunk آماده برای embedding (با متادیتا) تبدیل می‌کند."""
    chunks = []
    for i, section in enumerate(sections):
        text = section_to_text(section)
        if not text:
            continue

        has_table = any(b['type'] == 'table' for b in section['content_blocks'])

        if section['start_page'] == section['end_page']:
            page_range = str(section['start_page'])
        else:
            page_range = f"{section['start_page']}-{section['end_page']}"

        chunks.append({
            'chunk_id': i,
            'chapter': section['chapter'],
            'heading': section['heading'],
            'page_range': page_range,
            'has_table': has_table,
            'has_diagram_reference': section['has_diagram_reference'],
            'text': text,
            'char_count': len(text),
        })
    return chunks


def build_chunks(pdf_path: str) -> list[dict]:
    """تابع اصلی: مسیر PDF را می‌گیرد و لیست نهایی chunk ها را برمی‌گرداند."""
    doc = fitz.open(pdf_path)
    sections = build_sections(doc)
    merged_sections = merge_short_sections(sections)
    chunks = sections_to_chunks(merged_sections)
    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """chunk ها را در فایل JSON ذخیره می‌کند."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    default_pdf_path = project_root / 'manual.pdf'
    default_output_path = project_root / 'output_chunks_1.json'

    parser = argparse.ArgumentParser(description='Build and optionally save PDF chunks.')
    parser.add_argument('pdf_path', nargs='?', default=str(default_pdf_path))

    # --save
    parser.add_argument('--save', nargs='?', const=str(default_output_path),
                        dest='save_path', help='Save chunks using default or provided path')

    # -o / --output
    parser.add_argument('-o', '--output', dest='output_path',default=str(default_output_path),
                        help='Specify output file path')

    args = parser.parse_args()

    chunks = build_chunks(args.pdf_path)

    # انتخاب مسیر خروجی
    output_path = args.output_path or args.save_path 

    if output_path:
        save_chunks(chunks, output_path)        
        print(f'chunk ها در فایل ذخیره شدند: {args.output_path}')
        print()

    print(f'تعداد کل chunk های نهایی: {len(chunks)}')
    print()

    char_counts = [c['char_count'] for c in chunks]
    print(f'کوتاه‌ترین chunk: {min(char_counts)} کاراکتر')
    print(f'بلندترین chunk: {max(char_counts)} کاراکتر')
    print(f'میانگین: {sum(char_counts) // len(char_counts)} کاراکتر')
    print()

    print('=== نمونه‌ی ۵ chunk اول ===')
    for c in chunks[:5]:
        print(f"--- chunk {c['chunk_id']} | فصل {c['chapter']} | صفحه {c['page_range']} | جدول={c['has_table']} ---")
        print(c['text'][:200])
        print()

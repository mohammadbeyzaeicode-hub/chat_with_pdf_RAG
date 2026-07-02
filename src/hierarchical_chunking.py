"""
گام ۱۱ آموزش: بازطراحی chunking به معماری سلسله‌مراتبی Parent-Child.

این فایل جایگزین منطق ادغام section های کوتاه در chunking.py می‌شود.
دلیل بازطراحی (نتیجه‌ی یک بحث آموزشی عمیق): تابع merge_short_sections
در chunking.py، section های کوتاه و نامرتبط را فقط به‌خاطر کوتاه‌بودن
با هم ادغام می‌کرد (مثلاً «بدون قفل مرکزی» با «مخزن سوخت» و «پر کردن
باک بنزین» در یک chunk قاطی می‌شدند). این باعث می‌شد بردار embedding
آن chunk، میانگین معنایی چند موضوع نامرتبط باشد — که نه برای موضوع
اول مفید است، نه برای دوم، نه سوم.

راه‌حل: معماری Parent-Child این تناقض را با جداکردن دو نقش حل می‌کند:
  - Parent = هر بخش کامل (یک heading + تمام محتوای زیرش)، بدون هیچ
    ادغام مصنوعی با بخش‌های دیگر. هر Parent، یک واحد معنایی مستقل
    و کامل است.
  - Child = تکه‌های کوچک‌تر داخل یک Parent، که برای embedding و
    جستجوی معنایی دقیق استفاده می‌شوند. هر Child متن خودش را با
    عنوان Parent همراه می‌کند (context زمینه‌ای برای embedding بهتر).

هنگام پاسخ‌دهی: جستجو در سطح Child انجام می‌شود (دقیق)، اما به محض
پیدا شدن Child مرتبط، متن کامل Parent آن (نه فقط آن تکه‌ی کوچک) به
LLM داده می‌شود تا context کافی داشته باشد.

تصمیم‌های طراحی (گرفته‌شده در گفتگوی آموزشی):
  - تقسیم Parent به Child بر اساس مرز جمله (نه تعداد کاراکتر ثابت)،
    چون مرز جمله در فارسی با نقطه/علامت سوال مشخص است و بریدن وسط
    جمله معنا را خراب می‌کند.
  - جدول‌ها هرگز تقسیم نمی‌شوند؛ کل جدول یک Child مستقل می‌ماند،
    چون جدول یک واحد معنایی غیرقابل‌تفکیک است.
  - شماره صفحه‌ی تنها که گاهی وسط متن ظاهر می‌شد، در page_processor.py
    فیلتر شده (نه اینجا)، چون مشکل از مرحله‌ی استخراج می‌آمد.
"""

import json
from pathlib import Path
import re

import fitz

from page_processor import process_page
from structure_extraction import assign_chapter_to_pages

# هر چند جمله یک Child بسازیم. عدد ۲ انتخاب شده چون اکثر پاراگراف‌های
# این سند (دفترچه راهنمای خودرو) جمله‌های نسبتاً کوتاهی دارند؛ گروه‌بندی
# ۲ به ۲ باعث می‌شود هر Child هم به‌اندازه‌ی کافی context داشته باشد
# (نه فقط یک جمله‌ی تک که گاهی خیلی کوتاه است) و هم به‌اندازه‌ی کافی
# مشخص و تک‌موضوعی بماند (نه آن‌قدر بزرگ که چند ایده را با هم قاطی کند).
SENTENCES_PER_CHILD = 2

# الگوی تقسیم جمله برای فارسی: نقطه، علامت سوال، یا علامت تعجب،
# به‌شرط آن‌که بعدش فاصله یا پایان متن بیاید (تا اعداد اعشاری مثل
# '۵۰.۲' به اشتباه به دو جمله تقسیم نشوند)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.؟!])\s+')


def split_into_sentences(text: str) -> list[str]:
    """متن یک پاراگراف را به جمله‌ها تقسیم می‌کند."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_parents(doc: fitz.Document) -> list[dict]:
    """
    تمام صفحات سند را پردازش می‌کند و بلوک‌ها را به Parent ها گروه‌بندی
    می‌کند: هر Parent با یک heading شروع می‌شود و تا heading بعدی ادامه
    می‌یابد. برخلاف نسخه‌ی قبلی (chunking.py)، اینجا هیچ ادغامی بین
    Parent های مختلف انجام نمی‌شود — هر Parent، هرچقدر هم کوتاه، مستقل
    باقی می‌ماند.
    """
    page_to_chapter = assign_chapter_to_pages(doc)
    parents = []
    current = None

    for page in doc:
        page_idx = page.number
        chapter = page_to_chapter.get(page_idx)
        result = process_page(page)
        page_number = result['page_number']
        has_diagram = result['has_diagram_reference']

        for block in result['blocks']:
            if block['type'] == 'heading':
                if current is not None:
                    parents.append(current)
                current = {
                    'chapter': chapter,
                    'heading': block['text'],
                    'content_blocks': [],
                    'start_page': page_number,
                    'end_page': page_number,
                    'has_diagram_reference': has_diagram,
                }
            else:
                if current is None:
                    # محتوای قبل از اولین heading سند (جلد، مقدمه)
                    current = {
                        'chapter': chapter,
                        'heading': None,
                        'content_blocks': [],
                        'start_page': page_number,
                        'end_page': page_number,
                        'has_diagram_reference': has_diagram,
                    }
                current['content_blocks'].append(block)
                current['end_page'] = page_number
                current['has_diagram_reference'] = current['has_diagram_reference'] or has_diagram

    if current is not None:
        parents.append(current)

    return parents


def parent_full_text(parent: dict) -> str:
    """متن کامل یک Parent را می‌سازد: عنوان + تمام محتوای زیرش."""
    parts = []
    if parent['heading']:
        parts.append(parent['heading'])
    for block in parent['content_blocks']:
        if block['type'] == 'warning':
            parts.append(f"[اخطار: {block['text']}]")
        else:
            parts.append(block['text'])
    return '\n'.join(parts).strip()


def build_children(parent: dict, parent_id: str) -> list[dict]:
    """
    یک Parent را به چند Child کوچک‌تر تقسیم می‌کند.

    قوانین:
      - بلوک‌های نوع 'table' هرگز تقسیم نمی‌شوند؛ هر جدول یک Child مستقل است.
      - بلوک‌های 'body' و 'warning' به جمله تقسیم می‌شوند و هر
        SENTENCES_PER_CHILD جمله یک Child می‌سازد.
      - متن هر Child برای embedding، با عنوان Parent همراه می‌شود
        (context زمینه‌ای) تا جستجوی معنایی دقیق‌تر باشد.
    """
    children = []
    heading = parent['heading'] or ''
    child_idx = 0

    # جمع‌آوری جمله‌های تمام بلوک‌های غیر-جدول این Parent
    pending_sentences = []

    def flush_pending():
        nonlocal child_idx, pending_sentences
        if not pending_sentences:
            return
        for i in range(0, len(pending_sentences), SENTENCES_PER_CHILD):
            group = pending_sentences[i:i + SENTENCES_PER_CHILD]
            raw_text = ' '.join(group)
            embedding_text = f'{heading}\n{raw_text}'.strip() if heading else raw_text
            children.append({
                'child_id': f'{parent_id}_c{child_idx}',
                'parent_id': parent_id,
                'raw_text': raw_text,
                'embedding_text': embedding_text,
                'is_table': False,
            })
            child_idx += 1
        pending_sentences = []

    for block in parent['content_blocks']:
        if block['type'] == 'table':
            # قبل از پردازش جدول، هر جمله‌ی در صف مانده را Child کن
            flush_pending()
            embedding_text = f'{heading}\n{block["text"]}'.strip() if heading else block['text']
            children.append({
                'child_id': f'{parent_id}_c{child_idx}',
                'parent_id': parent_id,
                'raw_text': block['text'],
                'embedding_text': embedding_text,
                'is_table': True,
            })
            child_idx += 1
        else:
            prefix = '[اخطار] ' if block['type'] == 'warning' else ''
            sentences = split_into_sentences(block['text'])
            pending_sentences.extend(f'{prefix}{s}' if prefix else s for s in sentences)

    flush_pending()

    # اگر Parent هیچ محتوایی نداشت (فقط heading، بدون body/table) —
    # مثل صفحات کاملاً وکتوری که در گام ۶ آموزش دیدیم — حداقل یک Child
    # فقط با عنوان می‌سازیم تا این بخش گم نشود
    if not children and heading:
        children.append({
            'child_id': f'{parent_id}_c0',
            'parent_id': parent_id,
            'raw_text': heading,
            'embedding_text': heading,
            'is_table': False,
        })

    return children


def build_hierarchical_chunks(pdf_path: str) -> dict:
    """
    تابع اصلی: مسیر PDF را می‌گیرد و ساختار کامل Parent-Child را
    برمی‌گرداند.

    خروجی: دیکشنری با دو کلید:
      'parents': لیست Parent ها (هر کدام شامل متن کامل و متادیتا)
      'children': لیست Child ها (هر کدام شامل embedding_text و parent_id)

    نکته‌ی شناخته‌شده (گام ۱۱ آموزش): حدود ۲۱٪ از Parent های ساخته‌شده
    متن بسیار کوتاهی دارند (زیر ۳۰ کاراکتر)، که عمدتاً از دو منبع می‌آیند:
      ۱. صفحه‌ی جلد کتاب (سه گزینه‌ی گیربکس که کنار هم چیده شده‌اند)
      ۲. صفحه‌ی فهرست مطالب اصلی (که هر نام فصل، به‌خاطر فونت Bold،
         به‌اشتباه یک heading مستقل با Parent جداگانه تشخیص داده می‌شود)
    تصمیم گرفته شد این Parent ها فعلاً فیلتر *نشوند*، تا با کلید API
    واقعی تأثیر واقعی‌شان روی کیفیت نتایج جستجو سنجیده شود، به‌جای
    حذف زودهنگام بر اساس فرض. اگر در عمل این موارد نتایج جستجو را
    آلوده کردند (مثلاً جستجوی نامرتبط با کلمه‌ی پرتکرار 'پژو' بالا
    بیاید)، باید اینجا یک فیلتر طول حداقلی (مثلاً >= 20 کاراکتر)
    قبل از بازگرداندن parents_out/children_out اضافه شود.
    """
    doc = fitz.open(pdf_path)
    raw_parents = build_parents(doc)

    parents_out = []
    children_out = []

    for i, p in enumerate(raw_parents):
        parent_id = f'parent_{i}'
        full_text = parent_full_text(p)
        if not full_text:
            continue

        if p['start_page'] == p['end_page']:
            page_range = str(p['start_page'])
        else:
            page_range = f"{p['start_page']}-{p['end_page']}"

        parents_out.append({
            'parent_id': parent_id,
            'chapter': p['chapter'],
            'heading': p['heading'],
            'page_range': page_range,
            'has_diagram_reference': p['has_diagram_reference'],
            'text': full_text,
        })

        children_out.extend(build_children(p, parent_id))

    return {'parents': parents_out, 'children': children_out}


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]      
    result = build_hierarchical_chunks(f'{project_root}/manual.pdf')
    parents = result['parents']
    children = result['children']

    print(f'تعداد Parent ها: {len(parents)}')
    print(f'تعداد Child ها: {len(children)}')
    print()

    # نمایش نمونه‌ی همان بخش مشکل‌دار قبلی (مخزن سوخت) برای مقایسه
    for p in parents:
        if p['heading'] and 'مخزن سوخت' in p['heading']:
            print(f"=== Parent: {p['heading']} (صفحه {p['page_range']}) ===")
            print(p['text'])
            print()
            print('--- Child های همین Parent ---')
            for c in children:
                if c['parent_id'] == p['parent_id']:
                    print(f"  [{'TABLE' if c['is_table'] else 'TEXT'}] {c['embedding_text'][:80]!r}")
            break

    with open(f'{project_root}/hierarchical_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print()
    print(f"ذخیره شد: {project_root}/hierarchical_chunks.json ({len(parents)} parent, {len(children)} child)")

"""
پردازش کامل و یکپارچه‌ی یک صفحه از PDF.

این ماژول، حاصل جمع‌بندی گام‌های ۱ تا ۵ آموزش است. سه ماژول قبلی را
با هم ترکیب می‌کند:
  - persian_text_utils      (گام ۲ و ۳: اصلاح RTL و تشخیص نمودار شماره‌گذاری‌شده)
  - structure_extraction    (گام ۴: تشخیص heading/body/warning/chapter)

و یک مشکل جدید که در همین گام (۵) کشف شد را هم حل می‌کند: تکرار محتوای
جدول در خروجی متن معمولی (با حذف بر اساس bbox).

خروجی نهایی هر صفحه، یک دیکشنری با کلیدهای زیر است:
    page_number:        شماره صفحه (1-based، برای نمایش به کاربر)
    blocks:              لیستی از بلوک‌های محتوا به ترتیب صفحه؛ هر بلوک یکی از:
                           {'type': 'heading', 'text': ...}
                           {'type': 'body', 'text': ...}
                           {'type': 'warning', 'text': ...}
                           {'type': 'table', 'markdown': ...}
    has_diagram_reference: آیا این صفحه به یک نمودار شماره‌گذاری‌شده وابسته است
"""

from pathlib import Path

import fitz

from persian_text_utils import clean_cell_for_markdown, page_has_numbered_diagram_reference
from structure_extraction import extract_structured_lines

# حداکثر نسبت سلول‌های خالی که هنوز یک جدول را «واقعی» در نظر می‌گیریم.
# در گام ۷ آموزش (chunking) کشف شد که find_tables() گاهی متن جلد یا
# نمودارهای پراکنده را به اشتباه جدول تشخیص می‌دهد. این جدول‌های کاذب
# همگی نسبت سلول خالی بسیار بالایی (۸۳٪ تا ۹۴٪) داشتند، در حالی که
# جدول‌های واقعی سند (مثل جدول صندلی کودک) نسبت خالی پایینی دارند.
MAX_EMPTY_CELL_RATIO = 0.5


def table_to_markdown(rows) -> str:
    """یک جدول (لیست سطرها) را به فرمت Markdown table تبدیل می‌کند."""
    if not rows:
        return ''
    header = rows[0]
    body = rows[1:]
    md_lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for row in body:
        md_lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(md_lines)


def is_likely_real_table(rows) -> bool:
    """
    تشخیص می‌دهد که آیا یک جدول استخراج‌شده واقعی است یا یک false
    positive از find_tables() (که گاهی متن جلد یا نمودارهای پراکنده
    را با جدول اشتباه می‌گیرد).

    معیار: نسبت سلول‌های خالی/None باید کمتر از MAX_EMPTY_CELL_RATIO باشد.
    """
    total_cells = sum(len(r) for r in rows)
    if total_cells == 0:
        return False
    empty_cells = sum(1 for r in rows for c in r if not c or not str(c).strip())
    return (empty_cells / total_cells) <= MAX_EMPTY_CELL_RATIO


def process_page(page: fitz.Page) -> dict:
    """
    یک صفحه‌ی PDF را به‌طور کامل پردازش می‌کند: جدول‌ها، متن ساختاریافته،
    و تشخیص وابستگی به نمودار را با هم ترکیب می‌کند.
    """
    # ۱. ابتدا جدول‌ها را استخراج می‌کنیم تا bbox آن‌ها را برای فیلتر متن داشته باشیم
    #    فقط جدول‌هایی که از فیلتر کیفیت رد می‌شوند (is_likely_real_table) در نظر
    #    گرفته می‌شوند؛ جدول‌های کاذب نادیده گرفته می‌شوند و متن خامشان همچنان
    #    از مسیر عادی استخراج متن (مرحله‌ی ۲) عبور می‌کند.
    found_tables = page.find_tables()
    table_bboxes = []
    table_markdowns = []

    for tab in found_tables.tables:
        raw_rows = tab.extract()
        if not is_likely_real_table(raw_rows):
            continue

        table_bboxes.append(tab.bbox)
        cleaned_rows = [
            [clean_cell_for_markdown(cell) for cell in row]
            for row in raw_rows
        ]
        table_markdowns.append({
            'bbox': tab.bbox,
            'markdown': table_to_markdown(cleaned_rows),
        })

    # ۲. خطوط متنی را استخراج می‌کنیم، با حذف نواحی جدول (تا تکرار نشود)
    structured_lines = extract_structured_lines(page, exclude_bboxes=table_bboxes)

    # ۳. بلوک‌های متنی متوالی هم‌نوع را به هم می‌چسبانیم
    #    (چون خطوط شکسته‌شده‌ی یک پاراگراف، در گام ۴ به‌صورت چندین خط body جدا آمدند)
    blocks = []
    for line in structured_lines:
        text = line['text']
        line_type = line['line_type']

        # شماره‌ی فصل تنها (مثل '1') برای محتوای نهایی مفید نیست، حذفش می‌کنیم
        if line_type == 'chapter_number':
            continue

        # شماره‌ی صفحه‌ی چاپی تنها (مثل '21') که گاهی به‌صورت یک بلوک body
        # کاملاً مجزا در وسط محتوای صفحه ظاهر می‌شود (نه در حاشیه‌ی استاندارد
        # بالا/پایین که قبلاً فیلتر کرده‌ایم). این عدد هیچ ارزش معنایی برای
        # embedding ندارد و فقط نویز اضافه می‌کند. تشخیص: متن body که، پس از
        # حذف فاصله، فقط از رقم (فارسی یا لاتین) تشکیل شده و کوتاه است
        # (صفحات این سند حداکثر ۳ رقمی‌اند).
        if line_type == 'body':
            digits_only = text.strip().translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
            if digits_only.isdigit() and len(digits_only) <= 3:
                continue

        mapped_type = {'heading': 'heading', 'warning_label': 'warning', 'body': 'body'}[line_type]

        if blocks and blocks[-1]['type'] == mapped_type and mapped_type == 'body':
            # چسباندن خطوط body متوالی به هم به‌عنوان یک پاراگراف واحد
            blocks[-1]['text'] += ' ' + text
        else:
            blocks.append({'type': mapped_type, 'text': text})

    # ۴. جدول‌ها را در موقعیت تقریبی صحیح (بر اساس bbox عمودی) درون لیست بلوک‌ها قرار می‌دهیم
    #    برای سادگی در این فاز پروژه، جدول‌ها را در همان ترتیبی که پیدا شدند
    #    در انتهای بلوک‌های متنی صفحه قرار می‌دهیم. اگر بعداً نیاز به ترتیب
    #    دقیق‌تر بود (جدول وسط متن)، باید بر اساس مقایسه‌ی bbox.y0 مرتب‌سازی شود.
    for tm in table_markdowns:
        blocks.append({'type': 'table', 'text': tm['markdown']})

    full_text_for_diagram_check = page.get_text()

    return {
        'page_number': page.number + 1,  # تبدیل به 1-based برای نمایش به کاربر
        'blocks': blocks,
        'has_diagram_reference': page_has_numbered_diagram_reference(full_text_for_diagram_check),
    }


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    doc = fitz.open(f'{project_root}/manual.pdf')

    for page_idx in [9, 11, 40]:
        page = doc[page_idx]
        result = process_page(page)
        print(f"=== صفحه چاپی {result['page_number']} (دارای ارجاع به نمودار: {result['has_diagram_reference']}) ===")
        for b in result['blocks']:
            preview = b['text'][:70].replace('\n', ' ⏎ ')
            print(f"  [{b['type']:8s}] {preview}")
        print()

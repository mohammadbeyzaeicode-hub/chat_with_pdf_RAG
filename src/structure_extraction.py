"""
استخراج بلوک‌های ساختاریافته از هر صفحه‌ی PDF.

در گام ۴ آموزش، با بررسی متادیتای فونت صفحات نمونه، این الگو پیدا شد:

  - متن معمولی          -> فونت 'BNazanin' (بدون Bold)، سایز ~10
  - عنوان فصل (header)   -> فونت 'BNazaninBold'، سایز ~10، در بالاترین خط صفحه
  - شماره فصل بزرگ       -> فونت 'BNazaninBold'، سایز ~21 (خیلی بزرگ‌تر از بقیه)
  - عنوان زیربخش         -> فونت 'BNazaninBold'، سایز ~10، کوتاه، در ابتدای یک بلوک محتوای جدید
  - برچسب هشدار ایمنی    -> فونت 'BNazaninBold'، سایز ~10، اما متن دقیقاً یکی از
                            کلمات کلیدی ثابت است: اخطار / هشدار / توجه / احتیاط

نکته‌ی مهم: عنوان زیربخش و برچسب هشدار از نظر فونت/سایز کاملاً یکسان‌اند؛
تنها راه تشخیص‌شان از هم، مقایسه‌ی متن با مجموعه‌ی کلمات کلیدی ثابت هشدار است.
این موضوع را در گام ۴ با بررسی فراوانی متن‌های Bold کوتاه در کل سند کشف کردیم.
"""

import fitz

WARNING_LABELS = {'اخطار', 'هشدار', 'توجه', 'احتیاط'}

# نسبت ارتفاعی (از بالای صفحه) که در آن، هر متنی را «هدر تکراری صفحه»
# در نظر می‌گیریم، نه عنوان واقعی محتوا. این هدر همان نام فصل کلی است
# که در بالای هر صفحه‌ی آن فصل تکرار می‌شود (مثل 'خودروی شما در یک نگاه').
#
# کشف‌شده در گام ۱۰ آموزش (بازبینی کیفیت بعد از تست واقعی RAG):
# یک نمونه‌ی واقعی نشان داد که هدر تکراری 'درها' در تمام صفحات فصل ۲
# دقیقاً در y نسبی=۰.۰۴۵ از بالای صفحه قرار دارد، در حالی که نزدیک‌ترین
# heading واقعی محتوا (مثل 'پر کردن باک بنزین') در y نسبی=۰.۱۱۰ است.
# قبل از این اصلاح، این هدر به‌عنوان یک heading واقعی پردازش می‌شد و
# باعث می‌شد در مرحله‌ی ادغام بخش‌های کوتاه (chunking.py)، دو موضوع
# کاملاً متفاوت (مثل 'بدون قفل مرکزی' و 'مخزن سوخت') در یک chunk با
# عنوان نادرست قاطی شوند.
HEADER_ZONE_RELATIVE_HEIGHT = 0.08

# اندازه‌ی فونت شماره‌ی فصل بزرگ، معمولاً خیلی بزرگ‌تر از متن عادی است
CHAPTER_NUMBER_SIZE_THRESHOLD = 15.0


def extract_structured_lines(page: fitz.Page, exclude_bboxes: list = None) -> list[dict]:
    """
    یک صفحه را به لیستی از خطوط ساختاریافته تبدیل می‌کند.
    هر آیتم شامل: text, font, size, is_bold, line_type, bbox

    line_type یکی از مقادیر زیر است:
        'chapter_number'   - شماره‌ی بزرگ فصل (مثل عدد تنها '1' با فونت بزرگ)
        'warning_label'    - برچسب هشدار ایمنی (اخطار/هشدار/توجه/احتیاط)
        'heading'          - عنوان بخش یا زیربخش (Bold، کوتاه، غیر از موارد بالا)
        'body'             - متن معمولی

    نکته‌ی پیاده‌سازی: پردازش در سطح "span" انجام می‌شود، نه کل خط.
    دلیل: در برخی صفحات (مثلاً وقتی شماره‌ی فصل و عنوان زیربخش بعد از آن
    در ادامه‌ی هم چیده شده‌اند)، PDF این دو را در یک خط فیزیکی واحد قرار
    می‌دهد، اما با span های جدا و سایزهای متفاوت. اگر در سطح خط تصمیم
    بگیریم، این دو با هم قاطی می‌شوند. این مشکل در گام ۴ آموزش با مقایسه‌ی
    صفحه ۹ (که ادغام شده بود) و صفحه ۱۱ (که جدا بود) کشف شد.

    پارامتر exclude_bboxes:
        لیستی از مستطیل‌های (x0, y0, x1, y1) که باید نادیده گرفته شوند.
        این برای جلوگیری از تکرار محتوای جدول استفاده می‌شود: همان متنی
        که از page.find_tables() به‌صورت Markdown استخراج می‌شود، اگر
        فیلتر نشود، دوباره به‌صورت خط‌های پاشیده و بی‌معنی (با heading
        اشتباه برای سرستون‌ها) در خروجی متن معمولی هم ظاهر می‌شود. این
        مشکل در گام ۵ آموزش، روی صفحه‌ی جدول صندلی کودک کشف شد.
    """
    exclude_bboxes = exclude_bboxes or []
    results = []
    blocks = page.get_text('dict')['blocks']
    page_height = page.rect.height
    header_zone_limit = page_height * HEADER_ZONE_RELATIVE_HEIGHT

    def bbox_inside_excluded(bbox):
        x0, y0, x1, y1 = bbox
        for ex0, ey0, ex1, ey1 in exclude_bboxes:
            # اگر مرکز این bbox داخل ناحیه‌ی حذف‌شده باشد، نادیده‌اش می‌گیریم
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                return True
        return False

    for b in blocks:
        if 'lines' not in b:
            continue
        for line in b['lines']:
            spans = line['spans']
            if not spans:
                continue

            line_bbox = line.get('bbox')
            if line_bbox and bbox_inside_excluded(line_bbox):
                continue

            # نادیده گرفتن خطوطی که در ناحیه‌ی هدر تکراری بالای صفحه قرار دارند
            # (مثل نام فصل که در بالای هر صفحه تکرار می‌شود). این خطوط محتوای
            # واقعی ندارند و اگر به‌عنوان heading در نظر گرفته شوند، باعث
            # قاطی‌شدن موضوعات نامرتبط در مرحله‌ی chunking می‌شوند.
            if line_bbox and line_bbox[1] < header_zone_limit:
                continue

            # گروه‌بندی span های متوالی که سایز/بولد یکسانی دارند،
            # تا تکه‌های ریز (مثل نیم‌فاصله‌ی تنها) با همسایه‌ی هم‌نوعشان ادغام شوند
            groups = []
            for s in spans:
                text = s['text']
                if not text:
                    continue
                size = round(s['size'], 1)
                is_bold = 'Bold' in s['font']
                font = s['font']

                if groups and groups[-1]['is_bold'] == is_bold and abs(groups[-1]['size'] - size) < 0.5:
                    groups[-1]['text'] += text
                else:
                    groups.append({'text': text, 'size': size, 'is_bold': is_bold, 'font': font})

            for g in groups:
                line_text = g['text'].strip()
                if not line_text:
                    continue
                if 'cargeek.ir' in line_text:
                    continue

                size = g['size']
                is_bold = g['is_bold']

                # آیا متن واقعا فقط شامل رقم است؟ (مثل '1' یا '۵')
                # هم رقم لاتین و هم فارسی را در نظر می‌گیریم
                is_pure_digit = line_text.translate(
                    str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
                ).isdigit()

                # آیا این یک عنوان معنادار است؟ باید حداقل دو حرف الفبایی واقعی داشته باشد
                # (نه فقط علائم نگارشی مثل '-' یا تنها یک نقطه)
                meaningful_char_count = sum(
                    1 for c in line_text if c.isalnum()
                )

                if is_bold and size >= CHAPTER_NUMBER_SIZE_THRESHOLD and is_pure_digit:
                    line_type = 'chapter_number'
                elif is_bold and line_text in WARNING_LABELS:
                    line_type = 'warning_label'
                elif is_bold and len(line_text) < 40 and meaningful_char_count >= 3:
                    line_type = 'heading'
                else:
                    line_type = 'body'

                results.append({
                    'text': line_text,
                    'font': g['font'],
                    'size': size,
                    'is_bold': is_bold,
                    'line_type': line_type,
                    'bbox': line_bbox,
                })

    return results


def find_chapter_start_pages(doc: fitz.Document, size_threshold: float = 40.0) -> dict:
    """
    صفحات شروع هر فصل را پیدا می‌کند.

    در گام ۷ آموزش (chunking) کشف شد که شماره‌ی فصل در صفحه‌ی شروع هر
    فصل با یک فونت بسیار بزرگ (~83) چاپ می‌شود — متفاوت از شماره‌ی
    فصل تکراری در هدر تمام صفحات (~21) که با تابع
    extract_structured_lines به‌عنوان 'chapter_number' شناسایی می‌شود.

    این دو اندازه فونت آن‌قدر متفاوت‌اند (83 در مقابل 21) که استفاده
    از یک threshold بالاتر (40) به‌جای آستانه‌ی قبلی، صفحات شروع فصل
    را با دقت کامل (بدون false positive) شناسایی می‌کند.

    خروجی: دیکشنری {شماره_فصل: page_index}
    """
    chapter_starts = {}
    for i, page in enumerate(doc):
        blocks = page.get_text('dict')['blocks']
        max_size = 0
        max_text = ''
        for b in blocks:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    if 'Bold' in span['font'] and span['size'] > max_size:
                        max_size = span['size']
                        max_text = span['text'].strip()

        if max_size >= size_threshold:
            digits = max_text.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
            if digits.isdigit():
                chapter_starts[int(digits)] = i

    return chapter_starts


def assign_chapter_to_pages(doc: fitz.Document) -> dict:
    """
    بر اساس صفحات شروع فصل، به هر صفحه‌ی سند یک شماره فصل نسبت می‌دهد.

    خروجی: دیکشنری {page_index: شماره_فصل}
    """
    chapter_starts = find_chapter_start_pages(doc)
    # مرتب‌سازی بر اساس شماره صفحه برای پیمایش ترتیبی
    sorted_starts = sorted(chapter_starts.items(), key=lambda kv: kv[1])

    page_to_chapter = {}
    for i in range(len(doc)):
        current_chapter = None
        for chapter_num, start_page in sorted_starts:
            if i >= start_page:
                current_chapter = chapter_num
            else:
                break
        page_to_chapter[i] = current_chapter

    return page_to_chapter


if __name__ == '__main__':
    doc = fitz.open('../manual.pdf')
    for page_idx in [9, 11]:
        print(f'=== صفحه index {page_idx} ===')
        lines = extract_structured_lines(doc[page_idx])
        for l in lines:
            print(f"[{l['line_type']:15s}] {l['text']}")
        print()

"""
تست گام ۲: استخراج جدول از یک صفحه‌ی مشخص PDF و اصلاح متن فارسی آن.

این اسکریپت صرفاً برای تأیید درست کار کردن pipeline تا همین مرحله است.
در مراحل بعدی، این منطق به یک ماژول استخراج کامل‌تر منتقل می‌شود.
"""

import sys
import fitz  # PyMuPDF

from persian_text_utils import clean_cell_for_markdown


def extract_table_from_page(pdf_path: str, page_index: int):
    """
    جدول‌های یک صفحه‌ی مشخص را با PyMuPDF پیدا و استخراج می‌کند،
    سپس هر سلول را با اصلاح RTL تمیز می‌کند.

    خروجی: لیستی از جدول‌ها؛ هر جدول یک لیست از سطرها است.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    tabs = page.find_tables()
    cleaned_tables = []

    for tab in tabs.tables:
        raw_rows = tab.extract()
        cleaned_rows = [
            [clean_cell_for_markdown(cell) for cell in row]
            for row in raw_rows
        ]
        cleaned_tables.append(cleaned_rows)

    return cleaned_tables


def table_to_markdown(rows) -> str:
    """یک جدول (لیست سطرها) را به فرمت Markdown table تبدیل می‌کند."""
    if not rows:
        return ''

    header = rows[0]
    body = rows[1:]

    md_lines = []
    md_lines.append('| ' + ' | '.join(header) + ' |')
    md_lines.append('|' + '|'.join(['---'] * len(header)) + '|')
    for row in body:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(md_lines)


if __name__ == '__main__':
    pdf_path = '../manual.pdf'  # فایل اصلی PDF در ریشه‌ی پروژه نگه‌داری می‌شود
    page_index = 40  # صفحه‌ی جدول صندلی کودک (پیدا شده در گام ۲)

    tables = extract_table_from_page(pdf_path, page_index)
    print(f'تعداد جدول پیدا شده در صفحه {page_index}: {len(tables)}\n')

    for i, table in enumerate(tables):
        print(f'=== جدول {i} (به فرمت Markdown) ===\n')
        print(table_to_markdown(table))
        print()

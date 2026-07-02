"""
گام ۱۱ آموزش (به‌روزرسانی): خواندن hierarchical_chunks.json (خروجی
hierarchical_chunking.py)، embed کردن فقط Child ها (نه Parent ها)،
و ذخیره‌ی نتیجه به‌همراه بردارهای embedding.

چرا فقط Child ها embed می‌شوند، نه Parent ها؟
طبق معماری Parent-Child که در گفتگوی آموزشی طراحی شد: Child ها واحد
جستجو هستند (کوچک و تک‌موضوعی، برای دقت بهتر embedding)، در حالی که
Parent ها فقط برای دادن context کامل به LLM در زمان پاسخ‌دهی استفاده
می‌شوند و خودشان مستقیماً جستجو نمی‌شوند. پس فقط هزینه‌ی embedding برای
Child ها پرداخت می‌شود (که عملاً ارزان‌تر هم هست، چون تعدادشان معمولاً
نزدیک یا کمی بیشتر از تعداد Parent هاست، نه چند برابر).

نکته: متنی که برای هر Child embed می‌شود، 'embedding_text' است
(که در hierarchical_chunking.py ساخته شده و عنوان Parent را هم به
متن خام Child اضافه می‌کند)، نه 'raw_text'.
"""

import json
from pathlib import Path
import time

from embedding import get_default_embedder


def embed_hierarchical_chunks(input_path: str, output_path: str, batch_log: bool = True) -> None:
    """
    فایل hierarchical_chunks.json را می‌خواند، فقط بخش 'children' را
    embed می‌کند (با استفاده از 'embedding_text')، و کل ساختار
    (parents + children با embedding) را در output_path ذخیره می‌کند.
    """
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    parents = data['parents']
    children = data['children']

    embedder = get_default_embedder()
    texts = [c['embedding_text'] for c in children]

    if batch_log:
        print(f'در حال embed کردن {len(texts)} Child با مدل {embedder.model_name} ...')

    start_time = time.time()
    vectors = embedder.embed_texts(texts)
    elapsed = time.time() - start_time

    if batch_log:
        print(f'انجام شد در {elapsed:.1f} ثانیه.')

    for child, vector in zip(children, vectors):
        child['embedding'] = vector
        child['embedding_model'] = embedder.model_name

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'parents': parents, 'children': children}, f, ensure_ascii=False)
        # بدون indent، چون بردارهای عددی حجم فایل را با indent بسیار بزرگ می‌کنند

    print(f'ذخیره شد: {output_path} ({len(parents)} parent, {len(children)} child embed‌شده، ابعاد بردار={embedder.dimensions})')


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    
    embed_hierarchical_chunks(f'{project_root}/hierarchical_chunks.json', f'{project_root}/embedded_hierarchical_chunks.json')

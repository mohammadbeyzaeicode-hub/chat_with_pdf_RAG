"""
گام ۱۱ آموزش (به‌روزرسانی): حلقه‌ی نهایی Chat-with-PDF با معماری
Parent-Child — جستجوی معنایی در سطح Child (دقیق و تک‌موضوعی)، اما
ساخت پاسخ نهایی با متن کامل Parent (context کافی برای LLM).

چرا این تغییر لازم بود: نسخه‌ی قبلی این فایل، مستقیماً از متن chunk
بازیابی‌شده استفاده می‌کرد. اما در chunking قدیمی، گاهی چند بخش
نامرتبط برای رسیدن به حداقل طول لازم با هم قاطی می‌شدند (مثل «بدون
قفل مرکزی» با «مخزن سوخت»)، که باعث می‌شد متادیتای heading گمراه‌کننده
باشد. با معماری Parent-Child، هر Parent مستقل و تک‌موضوعی است، و
جستجو روی Child های کوچک‌تر (که context عنوان را هم دارند) انجام
می‌شود، اما برای پاسخ‌دهی نهایی، متن کامل Parent (نه فقط همان Child)
به LLM داده می‌شود.

مثل embedding.py و vector_store.py، اینجا هم یک interface قابل‌تغییر
برای LLM تعریف شده (BaseLLM) تا اگر بعداً خواستیم از GPT-4o-mini به
Claude یا مدل دیگری عوض کنیم، فقط یک کلاس جدید اضافه می‌شود.

طراحی پرامپت (system prompt) سه اصل را رعایت می‌کند که در گفتگوی
آموزشی روی آن تصمیم گرفته شد:
  ۱. پاسخ فقط بر اساس context داده‌شده باشد، نه دانش عمومی مدل
     (چون دانش عمومی درباره‌ی یک مدل خاص خودرو می‌تواند نادرست باشد)
  ۲. اگر جواب در context نبود، صادقانه اعلام شود
  ۳. شماره صفحه‌ی منبع همیشه ذکر شود، و اگر Parent به نمودار تصویری
     وابسته بود (has_diagram_reference)، کاربر به دیدن خود تصویر در
     آن صفحه ارجاع داده شود
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

from openai import OpenAI
from config import client
from embedding import get_default_embedder
from vector_store import get_default_vector_store, load_parents_lookup


class BaseLLM(ABC):
    """Interface مشترک برای مدل‌های زبانی مسئول ساخت پاسخ نهایی."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        raise NotImplementedError


class OpenAIChatLLM(BaseLLM):
    """پیاده‌سازی با OpenAI Chat Completions API."""

    def __init__(self, model: str = 'gpt-4o-mini', api_key: str = None, temperature: float = 0.2):
        self._model = model
        self._temperature = temperature
        # self._client = OpenAI(api_key=api_key or os.environ.get('OPENAI_API_KEY'))
        self._client=client

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
        )
        return response.choices[0].message.content

_llm_instance = None
def get_default_llm() -> BaseLLM:
    """نقطه‌ی مرکزی انتخاب LLM پیش‌فرض پروژه. برای تغییر مدل، فقط همین تابع را ویرایش کنید."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OpenAIChatLLM(model='gpt-4o-mini')
    return _llm_instance


SYSTEM_PROMPT_TEMPLATE = """تو یک دستیار هوشمند هستی که فقط بر اساس دفترچه راهنمای مالک خودرو پژو ۲۰۷ به سوالات پاسخ می‌دهی.

قوانین مهم:
۱. فقط از اطلاعاتی که در «منابع» زیر آمده استفاده کن. هرگز از دانش عمومی خودت درباره‌ی خودروها استفاده نکن، چون ممکن است با این مدل خاص متفاوت باشد.
۲. اگر جواب سوال در منابع زیر پیدا نشد، صادقانه بگو: «این اطلاعات را در دفترچه راهنما پیدا نکردم.» هرگز پاسخ را از خودت نساز.
۳. در پایان پاسخ، همیشه شماره صفحه‌ی منبع (یا منابع) را ذکر کن. مثلاً: «(صفحه ۴۱)»
۴. اگر منبعی که از آن استفاده کردی علامت [نیازمند بررسی تصویر] دارد، حتماً به کاربر بگو که برای اطمینان کامل باید خود تصویر/نمودار آن صفحه را هم ببیند، چون بخشی از اطلاعات (مثل شماره‌گذاری اجزا) فقط در تصویر آمده و این‌جا در دسترس نیست.
۵. پاسخ را به زبان فارسی، روان و خلاصه بنویس.

منابع:
{context}
"""


def format_context(parent_list: list[dict]) -> str:
    """
    لیست Parent های یکتا (بعد از حذف تکرار) را به یک متن قابل‌فهم
    برای LLM تبدیل می‌کند، شامل متادیتای مهم (صفحه، و پرچم وابستگی
    به نمودار).
    """
    parts = []
    for i, p in enumerate(parent_list, start=1):
        diagram_note = ' [نیازمند بررسی تصویر]' if p.get('has_diagram_reference') else ''
        parts.append(
            f"--- منبع {i} (صفحه {p['page_range']}){diagram_note} ---\n{p['text']}"
        )
    return '\n\n'.join(parts)


def retrieve_parents(question: str, top_k: int = 4, where: dict = None,embedder=None, store=None, parents_lookup=None) -> list[dict]:
    """
    سوال را embed می‌کند، در سطح Child جستجو می‌کند (دقیق)، سپس برای
    هر Child پیدا‌شده، Parent کامل آن را برمی‌گرداند.

    نکته‌ی مهم: چند Child مختلف می‌توانند به یک Parent مشترک اشاره
    کنند (مثلاً دو جمله‌ی متفاوت از همان بخش «ایمنی کودکان»). در این
    صورت، آن Parent باید فقط یک‌بار در نتیجه‌ی نهایی ظاهر شود، نه
    تکراری — وگرنه هم در پرامپت LLM فضای اضافی هدر می‌رود و هم احتمال
    سردرگمی مدل بیشتر می‌شود. ترتیب بر اساس بهترین (کمترین distance)
    Child هر Parent حفظ می‌شود.
    """
    embedder =embedder or get_default_embedder()
    store =store or get_default_vector_store()
    
    project_root = Path(__file__).resolve().parents[1]
    
    parents_lookup =parents_lookup or load_parents_lookup(f'{project_root}/embedded_hierarchical_chunks.json')

    query_vector = embedder.embed_texts([question])[0]
    retrieved_children = store.query(query_vector, top_k=top_k, where=where)

    seen_parent_ids = set()
    unique_parents = []
    for child_result in retrieved_children:
        parent_id = child_result['metadata']['parent_id']
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        parent = parents_lookup.get(parent_id)
        if parent:
            unique_parents.append(parent)

    return unique_parents


def answer_question(question: str, top_k: int = 10, where: dict = None,embedder=None, store=None, parents_lookup=None, llm=None) -> dict:
    """
    تابع اصلی RAG: سوال کاربر را می‌گیرد، Child های مرتبط را پیدا
    می‌کند، Parent کامل هر کدام را بازیابی می‌کند، و پاسخ نهایی را
    با LLM می‌سازد.

    خروجی: دیکشنری شامل 'answer' (پاسخ نهایی) و 'sources' (Parent های
    استفاده‌شده، برای نمایش شفاف منبع به کاربر).
    """
    llm =llm or get_default_llm()

    unique_parents = retrieve_parents(question, top_k=top_k, where=where,embedder=embedder, store=store, parents_lookup=parents_lookup)

    context = format_context(unique_parents)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    answer = llm.generate(system_prompt=system_prompt, user_message=question)

    return {
        'answer': answer,
        'sources': unique_parents,
    }

class RAGPipeline:
    def __init__(self, embedded_chunks_path: str = '../embedded_hierarchical_chunks.json'):
        print('در حال راه‌اندازی RAG Pipeline...')
        self._embedder = get_default_embedder()
        self._store = get_default_vector_store()
        self._parents_lookup = load_parents_lookup(embedded_chunks_path)
        self._llm = get_default_llm()
        print('راه‌اندازی کامل شد.\n')

    def ask(self, question: str, top_k: int = 4, where: dict = None) -> dict:
        return answer_question(
            question, top_k=top_k, where=where,
            embedder=self._embedder,
            store=self._store,
            parents_lookup=self._parents_lookup,
            llm=self._llm,
        )
if __name__ == '__main__':
    import sys
    project_root = Path(__file__).resolve().parents[1]

    # question = ' '.join(sys.argv[1:]) or 'ظرفیت باک بنزین چقدر است؟'
    # question = ' '.join(sys.argv[1:]) or 'ظرفیت مخزن سوخت چقدر است؟'
    question = ' '.join(sys.argv[1:]) or 'حجم باک چقدر است؟'
   
    print(f'سوال: {question}\n')

    pipeline = RAGPipeline()
    result = pipeline.ask(question)

    print('=== پاسخ ===')
    print(result['answer'])
    print()
    print('=== منابع استفاده‌شده ===')
    for p in result['sources']:
        print(f"  صفحه {p['page_range']} | {p['heading']}")

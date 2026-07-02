"""
گام ۹ آموزش: ذخیره و جستجوی برداری (vector store) برای chunk های embed‌شده.

طراحی این ماژول مشابه embedding.py است: یک interface مشترک (BaseVectorStore)
تعریف شده تا اگر بعداً نیاز به تغییر دیتابیس برداری بود (مثلاً مهاجرت از
ChromaDB به Qdrant در مقیاس بزرگ‌تر)، فقط یک کلاس جدید نوشته شود؛ کد
جستجوی پروژه (مرحله‌ی retrieval در chatbot نهایی) بدون تغییر باقی می‌ماند.

چرا ChromaDB (تصمیم‌گیری‌شده در گفتگوی آموزشی):
  - کاملاً embedded و local، بدون نیاز به سرور جدا (برخلاف Qdrant/Milvus
    که نیاز به Docker دارند) یا سرویس ابری (برخلاف Pinecone)
  - پشتیبانی بومی از متادیتا + فیلتر کردن بر اساس آن (مثل chapter,
    has_table) که دقیقاً با متادیتای غنی chunk های ما هم‌خوان است
  - برای مقیاس کوچک ما (۳۱۹ vector)، جستجوی خطی ساده‌ی Chroma به اندازه‌ی
    کافی سریع است؛ نیازی به الگوریتم‌های approximate-search سنگین FAISS
    یا دیتابیس‌های production-grade نیست
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb


class BaseVectorStore(ABC):
    """Interface مشترک برای ذخیره و جستجوی برداری."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]) -> None:
        """مجموعه‌ای از رکوردها (با بردار، متادیتا، و متن) را ذخیره می‌کند."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int = 5, where: dict = None) -> list[dict]:
        """
        نزدیک‌ترین رکوردها به query_embedding را برمی‌گرداند.
        where: فیلتر اختیاری روی متادیتا (مثل {'chapter': 2})
        خروجی: لیستی از دیکشنری {id, document, metadata, distance}
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """تعداد کل رکوردهای ذخیره‌شده را برمی‌گرداند."""
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """پیاده‌سازی vector store با ChromaDB (ذخیره‌ی دائمی روی دیسک، بدون نیاز به سرور)."""

    def __init__(self, persist_directory: str = '../chroma_db', collection_name: str = 'pejo207_manual'):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]) -> None:
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: list[float], top_k: int = 5, where: dict = None) -> list[dict]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        output = []
        for i in range(len(results['ids'][0])):
            output.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
            })
        return output

    def count(self) -> int:
        return self._collection.count()


def get_default_vector_store() -> BaseVectorStore:
    """
    نقطه‌ی مرکزی انتخاب vector store پیش‌فرض پروژه.
    برای تغییر در آینده (مثلاً مهاجرت به Qdrant)، فقط همین تابع را ویرایش کنید.
    """
    project_root = Path(__file__).resolve().parents[1]
    
    return ChromaVectorStore(persist_directory=f'{project_root}/chroma_db', collection_name='pejo207_manual')


def load_embedded_chunks_into_store(embedded_chunks_path: str, store: BaseVectorStore = None) -> BaseVectorStore:
    """
    فایل embedded_hierarchical_chunks.json (خروجی embed_chunks.py با
    معماری Parent-Child) را می‌خواند و فقط Child ها را در vector store
    ذخیره می‌کند (چون فقط Child ها embedding دارند و قرار است جستجو شوند).

    متادیتای هر رکورد شامل 'parent_id' است تا بعداً در مرحله‌ی پاسخ‌دهی
    (rag_pipeline.py) بتوان متن کامل Parent مربوطه را پیدا کرد.

    نکته: متادیتای ChromaDB فقط مقادیر ساده (str, int, float, bool) را
    قبول می‌کند، نه None یا dict تو در تو. مقادیر None (مثل chapter
    برای صفحات ابتدایی بدون فصل) به رشته‌ی خالی تبدیل می‌شوند.
    """
    store = store or get_default_vector_store()

    with open(embedded_chunks_path, encoding='utf-8') as f:
        data = json.load(f)

    children = data['children']

    ids = [c['child_id'] for c in children]
    embeddings = [c['embedding'] for c in children]
    documents = [c['raw_text'] for c in children]

    metadatas = []
    for c in children:
        metadatas.append({
            'parent_id': c['parent_id'],
            'is_table': c['is_table'],
        })

    store.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f'{len(children)} Child در vector store ذخیره شد. تعداد کل رکوردها: {store.count()}')
    return store


def load_parents_lookup(embedded_chunks_path: str) -> dict:
    """
    بخش 'parents' فایل embedded_hierarchical_chunks.json را به یک
    دیکشنری {parent_id: parent_dict} تبدیل می‌کند، برای lookup سریع
    در زمان پاسخ‌دهی (وقتی یک Child پیدا شد، باید Parent کامل آن را
    با یک lookup ساده، نه پیمایش لیست، پیدا کنیم).
    """
    with open(embedded_chunks_path, encoding='utf-8') as f:
        data = json.load(f)
    return {p['parent_id']: p for p in data['parents']}


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    
    store = load_embedded_chunks_into_store(f'{project_root}/embedded_hierarchical_chunks.json')

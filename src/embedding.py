"""
گام ۸ آموزش: تبدیل chunk های متنی (خروجی chunking.py) به بردارهای embedding.

طراحی این ماژول عمداً «قابل‌تغییر» (swappable) است: یک کلاس پایه‌ی
abstract به نام BaseEmbedder تعریف شده که هر پیاده‌سازی مشخص (مثل
OpenAI یا بعداً BGE-m3 محلی) باید از آن ارث‌بری کند. بقیه‌ی پروژه
(مثل ذخیره در vector database) فقط با این interface مشترک کار می‌کند،
نه مستقیم با OpenAI. این یعنی اگر بعداً خواستیم مدل را عوض کنیم
(مثلاً به یک مدل محلی روی CPU به‌خاطر کیفیت بهتر فارسی)، فقط کافی است
یک کلاس جدید بنویسیم که از BaseEmbedder ارث‌بری کند؛ هیچ کد دیگری
در پروژه نیاز به تغییر ندارد.

چرا با text-embedding-3-small شروع کردیم (تصمیم‌گیری‌شده در گفتگوی
آموزشی): برای شروع سریع و بدون نیاز به نصب/اجرای مدل سنگین روی
سخت‌افزار محدود (بدون GPU)؛ کیفیت فارسی این مدل به‌طور رسمی روی
benchmark تخصصی فارسی (FaMTEB) سنجیده نشده، اما برای راه‌اندازی
اولیه‌ی pipeline کافی است. مسیر ارتقا به BGE-m3 (که در FaMTEB
بالاترین امتیاز retrieval فارسی را دارد) برای فاز بعدی باز نگه
داشته شده است.
"""

import os
from abc import ABC, abstractmethod

from huggingface_hub import login
from openai import OpenAI
from config import client

class BaseEmbedder(ABC):
    """
    Interface مشترک برای همه‌ی embedder ها.

    هر پیاده‌سازی جدید (OpenAI، BGE-m3 محلی، Cohere، ...) باید این
    دو متد را پیاده‌سازی کند تا بقیه‌ی پروژه بدون تغییر با آن کار کند.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """لیستی از متن‌ها را می‌گیرد و لیستی از بردارهای embedding برمی‌گرداند."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """نام مدل، برای ذخیره در متادیتا (مفید هنگام تغییر مدل در آینده)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """ابعاد بردار خروجی این مدل."""
        raise NotImplementedError


class OpenAIEmbedder(BaseEmbedder):
    """
    پیاده‌سازی embedder با استفاده از OpenAI API.

    نیازمند متغیر محیطی OPENAI_API_KEY (یا پاس دادن مستقیم api_key).
    """

    # حداکثر تعداد متن در هر batch ارسالی به API
    # (OpenAI محدودیت دارد؛ ۱۰۰ یک مقدار امن و رایج است)
    BATCH_SIZE = 100

    def __init__(self, model: str = 'text-embedding-3-small', api_key: str = None):
        self._model = model
        # self._client = OpenAI(api_key=api_key or os.environ.get('OPENAI_API_KEY'))
        self._client=client
        # ابعاد بردار بر اساس مدل انتخابی (مقادیر رسمی OpenAI)
        self._dims_by_model = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
        }

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dims_by_model.get(self._model, 1536)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        متن‌ها را به‌صورت batch (دسته‌ای) به API می‌فرستد، نه یکی‌یکی،
        تا تعداد درخواست‌های API و در نتیجه تأخیر شبکه کمتر شود.
        """
        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            response = self._client.embeddings.create(model=self._model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings


class BGEEmbedder(BaseEmbedder):
    """
    پیاده‌سازی embedder با مدل محلی BGE-m3 (BAAI/bge-m3)، اجراشده با
    کتابخانه‌ی sentence-transformers، بدون نیاز به اینترنت یا API key
    در زمان اجرا (فقط بار اول برای دانلود مدل نیاز به اینترنت است).
 
    چرا این مدل (تصمیم‌گیری‌شده در گفتگوی آموزشی، بعد از یک یافته‌ی
    تجربی واقعی): در تست واقعی پروژه با text-embedding-3-small، سوال
    «ظرفیت باک بنزین چقدر است؟» نتوانست chunk صحیح (که فقط عبارت
    «مخزن سوخت» را داشت، نه «باک») را پیدا کند — یعنی مدل OpenAI
    نتوانست این دو مترادف فارسی را به‌خوبی معادل بفهمد. این دقیقاً
    همان ضعفی است که benchmark تخصصی فارسی FaMTEB هم نشان داده بود:
    BGE-m3 بالاترین امتیاز retrieval فارسی (۷۴.۵۶) را دارد، در حالی
    که مدل‌های OpenAI روی این benchmark حتی ارزیابی نشده‌اند.
 
    نکته‌ی نصب: نیاز به `pip install sentence-transformers` دارد.
    بار اول که اجرا شود، مدل (~2.2GB) از huggingface.co دانلود و
    cache می‌شود؛ اجراهای بعدی از cache محلی استفاده می‌کنند و به
    اینترنت نیاز ندارند.
 
    نکته‌ی سخت‌افزاری: روی CPU بدون GPU، embed کردن چند صد متن ممکن
    است چند دقیقه طول بکشد (یک‌بار، در مرحله‌ی ساخت دیتابیس)؛ اما
    embed کردن یک سوال تکی در زمان پاسخ‌دهی همچنان سریع است (معمولاً
    زیر یک ثانیه روی CPU معمولی).
    """
 
    def __init__(self, model: str = 'BAAI/bge-m3'):
        # import در داخل __init__ (نه بالای فایل) تا کسی که فقط از
        # OpenAIEmbedder استفاده می‌کند، مجبور به نصب sentence-transformers
        # نباشد — این کتابخانه (با torch و transformers) سنگین است.
        from huggingface_hub import login
        from config import hf_key
        login(hf_key)
        

        
        from sentence_transformers import SentenceTransformer
 
        self._model_name = model
        self._model = SentenceTransformer(model)
 
    @property
    def model_name(self) -> str:
        return self._model_name
 
    @property
    def dimensions(self) -> int:
        return self._model.get_sentence_embedding_dimension()
 
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        normalize_embeddings=True تنظیم شده چون BGE-m3 طبق مستندات
        رسمی‌اش برای محاسبه‌ی شباهت کسینوسی (که ChromaDB پیش‌فرض
        استفاده می‌کند) باید بردارهای نرمال‌شده داشته باشد.
        """
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
        )
        return embeddings.tolist()
    # def embed_documents(self, texts: list[str]) -> list[list[float]]:
    #     return self._model.encode(
    #         texts,
    #         normalize_embeddings=True,
    #     ).tolist()


    # def embed_query(self, query: str) -> list[float]:
    #     query = (
    #         "Represent this sentence for searching relevant passages: "
    #         + query
    #     )

    #     return self._model.encode(
    #         query,
    #         normalize_embeddings=True,
    #     ).tolist()

 
_embedder_instance = None
def get_default_embedder() -> BaseEmbedder:
    """
    نقطه‌ی مرکزی انتخاب مدل پیش‌فرض پروژه.

    برای تغییر مدل embedding در آینده (مثلاً به BGE-m3 محلی)،
    کافی است فقط همین تابع را ویرایش کنید؛ بقیه‌ی کد پروژه
    (مثل embed_chunks.py) بدون تغییر باقی می‌ماند.
    """
    global _embedder_instance

    if _embedder_instance is None:
        # _embedder_instance = OpenAIEmbedder(model='text-embedding-3-small')
        _embedder_instance = BGEEmbedder()
    # return OpenAIEmbedder(model='text-embedding-3-large')
    return BGEEmbedder()


if __name__ == '__main__':
    embedder = get_default_embedder()
    sample_texts = [
        'سلام، این یک متن آزمایشی فارسی است.',
        'ظرفیت باک بنزین پژو ۲۰۷ چقدر است؟',
    ]
    vectors = embedder.embed_texts(sample_texts)
    print(f'مدل: {embedder.model_name}')
    print(f'ابعاد بردار: {embedder.dimensions}')
    print(f'تعداد بردار ساخته‌شده: {len(vectors)}')
    print(f'طول هر بردار: {len(vectors[0])}')
    print(f'نمونه‌ی ۵ عدد اول بردار اول: {vectors[0][:5]}')


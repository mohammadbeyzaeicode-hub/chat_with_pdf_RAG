"""
FastAPI REST API برای سیستم Chat-with-PDF دفترچه راهنمای پژو ۲۰۷.

این API موازی با رابط Streamlit (app.py) وجود دارد:
  - app.py   → رابط بصری برای نمایش دمو
  - api.py   → REST API برای ادغام با هر frontend یا سرویس دیگه

Endpoints:
  GET  /health  → وضعیت سیستم
  POST /ask     → پرسش و پاسخ
  GET  /stats   → آمار سند

نحوه‌ی اجرا:
    cd src
    uvicorn api:app --reload

مستندات خودکار (Swagger UI):
    http://localhost:8000/docs

نحوه‌ی تست با curl:
    curl -X POST http://localhost:8000/ask \\
         -H "Content-Type: application/json" \\
         -d '{"question": "ظرفیت باک بنزین چقدر است؟"}'
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


# ─── مدل‌های Pydantic برای Request و Response ────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description='سوال کاربر به فارسی',
        examples=['ظرفیت باک بنزین چقدر است؟'],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description='تعداد بخش‌های سند برای جستجو (پیش‌فرض: ۴)',
    )


class SourceModel(BaseModel):
    heading: Optional[str] = Field(None, description='عنوان بخش')
    page_range: str = Field(..., description='شماره صفحه یا بازه‌ی صفحات')
    has_diagram_reference: bool = Field(
        False,
        description='آیا این بخش به یک نمودار تصویری وابسته است؟',
    )
    text: str = Field(..., description='متن کامل بخش مرتبط از سند')


class AskResponse(BaseModel):
    answer: str = Field(..., description='پاسخ نهایی تولیدشده توسط LLM')
    sources: list[SourceModel] = Field(
        ...,
        description='بخش‌های سند که برای ساخت پاسخ استفاده شده‌اند',
    )


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    total_chunks: int


class StatsResponse(BaseModel):
    total_chunks: int = Field(..., description='تعداد کل Child ها در vector store')
    embedding_model: str
    llm_model: str


# ─── راه‌اندازی Pipeline با lifespan ─────────────────────────────────────────
# lifespan (جایگزین @app.on_event که deprecated شده) یعنی Pipeline
# یک‌بار موقع شروع server ساخته می‌شه و تا پایان نگه داشته می‌شه —
# دقیقاً همون مشکلی که در evaluation کشف کردیم (بارگذاری مکرر مدل)

pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    project_root = Path(__file__).resolve().parents[1]
    
    pipeline = RAGPipeline(embedded_chunks_path=f"{project_root}/embedded_hierarchical_chunks.json")
   
    yield
    # cleanup در صورت نیاز می‌شه اینجا اضافه کرد


# ─── ساخت اپلیکیشن ────────────────────────────────────────────────────────────
app = FastAPI(
    title='دستیار دفترچه راهنمای پژو ۲۰۷',
    description="""
## Chat with PDF — RAG API

این API امکان پرسش و پاسخ هوشمند بر اساس محتوای **دفترچه راهنمای مالک پژو ۲۰۷** را فراهم می‌کند.

### معماری سیستم
- **استخراج و پردازش:** PyMuPDF + پردازش اختصاصی متن فارسی (RTL fix، حذف هدر تکراری)
- **Chunking:** معماری Parent-Child سلسله‌مراتبی
- **Embedding:** BGE-m3 (بهترین مدل در benchmark فارسی FaMTEB)
- **Vector Store:** ChromaDB
- **LLM:** GPT-4o-mini

### نکته
پاسخ‌ها **فقط** بر اساس محتوای دفترچه راهنما تولید می‌شوند.
اگر اطلاعاتی در سند موجود نباشد، سیستم صادقانه اعلام می‌کند.
    """,
    version='1.0.0',
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    '/health',
    response_model=HealthResponse,
    summary='بررسی وضعیت سیستم',
    tags=['System'],
)
async def health():
    """وضعیت سیستم و اطلاعات مدل‌های استفاده‌شده را برمی‌گرداند."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Pipeline هنوز آماده نشده')
    return HealthResponse(
        status='ok',
        embedding_model=pipeline._embedder.model_name,
        llm_model='gpt-4o-mini',
        total_chunks=pipeline._store.count(),
    )


@app.get(
    '/stats',
    response_model=StatsResponse,
    summary='آمار سند',
    tags=['System'],
)
async def stats():
    """آمار کلی سند (تعداد chunk، مدل‌های استفاده‌شده) را برمی‌گرداند."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Pipeline هنوز آماده نشده')
    return StatsResponse(
        total_chunks=pipeline._store.count(),
        embedding_model=pipeline._embedder.model_name,
        llm_model='gpt-4o-mini',
    )


@app.post(
    '/ask',
    response_model=AskResponse,
    summary='پرسش و پاسخ',
    tags=['RAG'],
)
async def ask(request: AskRequest):
    """
    یک سوال به فارسی دریافت می‌کند و پاسخ + منابع را برمی‌گرداند.

    سیستم در سطح Child جستجو می‌کند (دقت بالا) و متن کامل Parent
    مربوطه را به LLM می‌دهد (context کافی).
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Pipeline هنوز آماده نشده')

    try:
        result = pipeline.ask(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'خطا در پردازش: {str(e)}')

    sources = [
        SourceModel(
            heading=p.get('heading'),
            page_range=p['page_range'],
            has_diagram_reference=p.get('has_diagram_reference', False),
            text=p['text'],
        )
        for p in result['sources']
    ]

    return AskResponse(answer=result['answer'], sources=sources)

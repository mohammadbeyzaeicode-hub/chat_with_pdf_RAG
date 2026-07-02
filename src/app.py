# """
# رابط کاربری Streamlit برای سیستم Chat-with-PDF دفترچه راهنمای پژو ۲۰۷.

# طراحی:
#   - RTL کامل با CSS سفارشی (چون Streamlit پیش‌فرض LTR است)
#   - تاریخچه‌ی کامل مکالمه در session_state
#   - نمایش منابع (صفحه + عنوان) زیر هر پاسخ
#   - accordion برای نمایش متن کامل Parent (شفافیت سیستم)
#   - RAGPipeline یک‌بار در session_state ذخیره می‌شود (نه هر بار از صفر)

# نحوه‌ی اجرا:
#     cd src
#     streamlit run app.py
# """

# from pathlib import Path

# import streamlit as st

# # ─── تنظیمات صفحه (باید اولین دستور Streamlit باشد) ─────────────────────────
# st.set_page_config(
#     page_title='دستیار پژو ۲۰۷',
#     page_icon='🚗',
#     layout='centered',
#     initial_sidebar_state='expanded',
# )

# # ─── CSS سفارشی برای RTL و ظاهر فارسی ────────────────────────────────────────
# st.markdown("""
# <style>
# /* فونت و جهت کلی صفحه */
# html, body, [class*="css"] {
#     font-family: 'Vazirmatn', 'Tahoma', 'Arial', sans-serif;
#     direction: rtl;
#     text-align: right;
# }

# /* جعبه‌ی ورود متن */
# .stTextInput input, .stTextArea textarea, .stChatInput textarea {
#     direction: rtl;
#     text-align: right;
# }

# /* حباب‌های چت */
# .stChatMessage {
#     direction: rtl;
# }

# /* سایدبار */
# [data-testid="stSidebar"] {
#     direction: rtl;
#     text-align: right;
# }

# /* دکمه‌ها */
# .stButton button {
#     direction: rtl;
# }

# /* کارت منبع */
# .source-card {
#     background: #f0f4ff;
#     border-right: 4px solid #4a6cf7;
#     border-radius: 8px;
#     padding: 10px 14px;
#     margin: 6px 0;
#     font-size: 0.9em;
# }

# /* هشدار نمودار */
# .diagram-warning {
#     background: #fff8e6;
#     border-right: 4px solid #f59e0b;
#     border-radius: 8px;
#     padding: 8px 12px;
#     margin: 4px 0;
#     font-size: 0.85em;
# }
# </style>
# """, unsafe_allow_html=True)


# # ─── بارگذاری یک‌باره‌ی Pipeline ──────────────────────────────────────────────
# @st.cache_resource(show_spinner='در حال بارگذاری مدل...')
# def load_pipeline():
#     """
#     RAGPipeline را یک‌بار می‌سازد و در cache نگه می‌دارد.
#     @st.cache_resource یعنی حتی اگه کاربر صفحه را refresh کند،
#     مدل دوباره لود نمی‌شود — دقیقاً مثل session_state اما در سطح server.
#     """
#     import sys
#     import os
#     sys.path.insert(0, os.path.dirname(__file__))
#     from rag_pipeline import RAGPipeline
#     project_root = Path(__file__).resolve().parents[1]

#     return RAGPipeline(embedded_chunks_path=f"{project_root}/embedded_hierarchical_chunks.json")


# # ─── سایدبار ──────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.title('🚗 دستیار پژو ۲۰۷')
#     st.caption('سیستم پرسش و پاسخ هوشمند بر اساس دفترچه راهنمای مالک')
#     st.divider()

#     # اطلاعات سیستم
#     st.subheader('⚙️ اطلاعات سیستم')
#     pipeline = load_pipeline()
#     st.markdown(f"""
#     - **مدل جستجو:** `{pipeline._embedder.model_name}`
#     - **مدل پاسخ:** `GPT-4o-mini`
#     - **تعداد بخش‌های سند:** `{pipeline._store.count()}`
#     """)

#     st.divider()
#     st.subheader('📖 درباره‌ی سند')
#     st.markdown("""
#     این دستیار بر اساس **دفترچه راهنمای مالک پژو ۲۰۷** (۱۳۵ صفحه)
#     پاسخ می‌دهد و خارج از محتوای این سند جواب نمی‌سازد.
#     """)

#     st.divider()
#     # دکمه‌ی پاک‌کردن مکالمه
#     if st.button('🗑️ پاک کردن مکالمه', use_container_width=True):
#         st.session_state.messages = []
#         st.rerun()


# # ─── مقداردهی اولیه‌ی تاریخچه ────────────────────────────────────────────────
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # ─── عنوان اصلی ───────────────────────────────────────────────────────────────
# st.title('💬 چت با دفترچه راهنمای پژو ۲۰۷')
# st.caption('سوال خود را درباره‌ی خودرو بپرسید. پاسخ‌ها فقط از متن دفترچه راهنما استخراج می‌شوند.')

# # ─── تابع نمایش منابع ─────────────────────────────────────────────────────────
# def _render_sources(sources: list[dict]):
#     if not sources:
#         return

#     st.markdown('---')
#     st.markdown('**📚 منابع:**')

#     for i, parent in enumerate(sources):
#         diagram_note = ''
#         if parent.get('has_diagram_reference'):
#             diagram_note = ' ⚠️'

#         with st.expander(
#             f"صفحه {parent['page_range']} — {parent['heading'] or 'بدون عنوان'}{diagram_note}"
#         ):
#             if parent.get('has_diagram_reference'):
#                 st.markdown("""
#                 <div class="diagram-warning">
#                 ⚠️ این بخش به یک نمودار یا تصویر در سند اشاره دارد.
#                 برای اطمینان کامل، صفحه‌ی مربوطه را در دفترچه راهنما مشاهده کنید.
#                 </div>
#                 """, unsafe_allow_html=True)

#             st.markdown(parent['text'])


# # ─── نمایش تاریخچه‌ی مکالمه ──────────────────────────────────────────────────
# for msg in st.session_state.messages:
#     with st.chat_message(msg['role'], avatar='👤' if msg['role'] == 'user' else '🤖'):
#         st.markdown(msg['content'])

#         # نمایش منابع برای پاسخ‌های قبلی
#         if msg['role'] == 'assistant' and 'sources' in msg:
#             _render_sources(msg['sources'])


# # ─── ورودی سوال ───────────────────────────────────────────────────────────────
# if question := st.chat_input('سوال خود را بنویسید...'):

#     # نمایش سوال کاربر
#     st.session_state.messages.append({'role': 'user', 'content': question})
#     with st.chat_message('user', avatar='👤'):
#         st.markdown(question)

#     # پردازش و نمایش پاسخ
#     with st.chat_message('assistant', avatar='🤖'):
#         with st.spinner('در حال جستجو در دفترچه راهنما...'):
#             result = pipeline.ask(question)

#         answer = result['answer']
#         sources = result['sources']

#         st.markdown(answer)
#         _render_sources(sources)

#     # ذخیره در تاریخچه
#     st.session_state.messages.append({
#         'role': 'assistant',
#         'content': answer,
#         'sources': sources,
#     })







'PYEOF'
"""
رابط کاربری Streamlit برای سیستم Chat-with-PDF دفترچه راهنمای پژو ۲۰۷.

معماری:
    Streamlit (frontend) ──HTTP──▶ FastAPI (backend) ──▶ RAGPipeline

این معماری از اتصال مستقیم Streamlit به RAGPipeline جدا شده تا:
  ۱. هر frontend دیگری (React، موبایل) بتواند به همان API وصل شود
  ۲. authentication، rate limiting، و logging فقط در یک جا (FastAPI) مدیریت شود
  ۳. Streamlit فقط مسئول نمایش باشد، نه پردازش

نحوه‌ی اجرا:
    # ابتدا FastAPI را راه‌اندازی کن:
    cd src && uvicorn api:app --reload --port 8000

    # سپس در ترمینال جداگانه، Streamlit را:
    cd src && streamlit run app.py
"""

import requests
import streamlit as st

# ─── تنظیمات ─────────────────────────────────────────────────────────────────
API_BASE_URL = 'http://localhost:8000'

# ─── تنظیمات صفحه ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='دستیار پژو ۲۰۷',
    page_icon='🚗',
    layout='centered',
    initial_sidebar_state='expanded',
)

# ─── CSS سفارشی برای RTL ─────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Vazirmatn', 'Tahoma', 'Arial', sans-serif;
    direction: rtl;
    text-align: right;
}
.stTextInput input, .stTextArea textarea, .stChatInput textarea {
    direction: rtl;
    text-align: right;
}
.stChatMessage { direction: rtl; }
[data-testid="stSidebar"] { direction: rtl; text-align: right; }
.stButton button { direction: rtl; }
.diagram-warning {
    background: #fff8e6;
    border-right: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)


# ─── توابع ارتباط با API ──────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_health() -> dict | None:
    """
    وضعیت سیستم را از FastAPI می‌گیرد.
    ttl=30 یعنی هر ۳۰ ثانیه یک‌بار refresh می‌شود.
    """
    try:
        resp = requests.get(f'{API_BASE_URL}/health', timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def ask_api(question: str, top_k: int = 4) -> dict | None:
    """سوال را به FastAPI می‌فرستد و پاسخ را برمی‌گرداند."""
    try:
        resp = requests.post(
            f'{API_BASE_URL}/ask',
            json={'question': question, 'top_k': top_k},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error('❌ اتصال به API برقرار نشد. مطمئن شوید FastAPI در حال اجرا است: `uvicorn api:app --reload`')
        return None
    except requests.exceptions.Timeout:
        st.error('⏱️ زمان انتظار به پایان رسید. لطفاً دوباره تلاش کنید.')
        return None
    except Exception as e:
        st.error(f'خطای غیرمنتظره: {e}')
        return None


# ─── سایدبار ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🚗 دستیار پژو ۲۰۷')
    st.caption('سیستم پرسش و پاسخ هوشمند بر اساس دفترچه راهنمای مالک')
    st.divider()

    st.subheader('⚙️ وضعیت سیستم')
    health = fetch_health()
    if health:
        st.success('🟢 API آماده است')
        st.markdown(f"""
        - **مدل جستجو:** `{health['embedding_model'].split('/')[-1]}`
        - **مدل پاسخ:** `{health['llm_model']}`
        - **تعداد بخش‌های سند:** `{health['total_chunks']}`
        """)
    else:
        st.error('🔴 API در دسترس نیست')
        st.caption('ابتدا FastAPI را اجرا کنید:')
        st.code('uvicorn api:app --reload', language='bash')

    st.divider()
    st.subheader('📖 درباره‌ی سند')
    st.markdown("""
    این دستیار بر اساس **دفترچه راهنمای مالک پژو ۲۰۷** (۱۳۵ صفحه)
    پاسخ می‌دهد و خارج از محتوای این سند جواب نمی‌سازد.
    """)

    st.divider()
    if st.button('🗑️ پاک کردن مکالمه', use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── مقداردهی اولیه‌ی تاریخچه ───────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []


# ─── تابع نمایش منابع ────────────────────────────────────────────────────────
def _render_sources(sources: list[dict]):
    if not sources:
        return
    st.markdown('---')
    st.markdown('**📚 منابع:**')
    for parent in sources:
        diagram_note = ' ⚠️' if parent.get('has_diagram_reference') else ''
        with st.expander(
            f"صفحه {parent['page_range']} — {parent.get('heading') or 'بدون عنوان'}{diagram_note}"
        ):
            if parent.get('has_diagram_reference'):
                st.markdown("""
                <div class="diagram-warning">
                ⚠️ این بخش به یک نمودار یا تصویر در سند اشاره دارد.
                برای اطمینان کامل، صفحه‌ی مربوطه را در دفترچه راهنما مشاهده کنید.
                </div>
                """, unsafe_allow_html=True)
            st.markdown(parent['text'])


# ─── عنوان اصلی ──────────────────────────────────────────────────────────────
st.title('💬 چت با دفترچه راهنمای پژو ۲۰۷')
st.caption('سوال خود را درباره‌ی خودرو بپرسید. پاسخ‌ها فقط از متن دفترچه راهنما استخراج می‌شوند.')

# ─── نمایش تاریخچه‌ی مکالمه ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg['role'], avatar='👤' if msg['role'] == 'user' else '🤖'):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant' and 'sources' in msg:
            _render_sources(msg['sources'])

# ─── ورودی سوال ──────────────────────────────────────────────────────────────
if question := st.chat_input('سوال خود را بنویسید...'):

    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user', avatar='👤'):
        st.markdown(question)

    with st.chat_message('assistant', avatar='🤖'):
        with st.spinner('در حال جستجو در دفترچه راهنما...'):
            result = ask_api(question)

        if result:
            answer = result['answer']
            sources = result['sources']
            st.markdown(answer)
            _render_sources(sources)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': answer,
                'sources': sources,
            })
# PYEOF
# echo "✅ فایل نوشته شد"
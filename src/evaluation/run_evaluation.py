"""
گام ۱۲ آموزش: ارزیابی سیستماتیک کیفیت سیستم RAG.

چرا این مرحله مهم است: تا اینجا کیفیت سیستم را فقط با چند سوال
دستی و پراکنده سنجیده بودیم (مثل ماجرای «باک» در مقابل «مخزن سوخت»).
این روش برای کشف باگ خوب است، اما برای ادعای «سیستم خوب کار می‌کند»
در یک نمونه‌کار حرفه‌ای کافی نیست — نیاز به یک معیار عددی و
تکرارپذیر داریم.

این اسکریپت یک مجموعه سوال از پیش تعریف‌شده (eval_questions.json) را
اجرا می‌کند و دو معیار را می‌سنجد:
  ۱. Retrieval Accuracy: آیا صفحه‌ی مورد انتظار، در میان صفحات
     منابع بازگشتی هست؟ (یعنی آیا سیستم اصلاً جای درست را پیدا کرد)
  ۲. Answer Correctness (تقریبی): آیا کلمات کلیدی مورد انتظار در
     متن پاسخ نهایی LLM وجود دارند؟

نکته‌ی مهم درباره‌ی روش سنجش: این یک سنجش "تقریبی بر اساس کلیدواژه"
است، نه ارزیابی معنایی دقیق (که نیاز به یک LLM داور جداگانه دارد).
این روش ساده عمداً انتخاب شده چون: (الف) بدون هزینه‌ی API اضافه
قابل اجراست، (ب) برای فکت‌های کوتاه و دقیق (مثل اعداد و رویه‌ها)
که اکثر سوالات این مجموعه از همین نوع‌اند، قابل اعتماد است.
برای سوالات با category=out_of_scope، معیار موفقیت برعکس است:
موفقیت یعنی سیستم باید بگوید جواب را پیدا نکرده، نه این‌که چیزی
از خودش بسازد.
"""

import json


from rag_pipeline import answer_question


def page_in_expected(expected_page: str, source_pages: list[str]) -> bool:
    """
    بررسی می‌کند که آیا expected_page در میان صفحات منابع بازگشتی است.
    چون page_range می‌تواند بازه باشد (مثل '40-41')، تطابق هم برای
    صفحه‌ی تکی و هم برای همپوشانی بازه‌ها در نظر گرفته می‌شود.
    """
    if expected_page is None:
        return None  # سوالات out_of_scope صفحه‌ی موردانتظار ندارند

    def page_range_to_set(page_range: str) -> set:
        if '-' in page_range:
            start, end = page_range.split('-')
            return set(range(int(start), int(end) + 1))
        return {int(page_range)}

    expected_set = page_range_to_set(expected_page)
    for sp in source_pages:
        if expected_set & page_range_to_set(sp):
            return True
    return False


def keywords_in_answer(expected_keywords: list[str], answer: str) -> bool:
    """بررسی می‌کند که آیا همه‌ی کلیدواژه‌های موردانتظار در پاسخ هستند."""
    if not expected_keywords:
        return None
    return all(kw in answer for kw in expected_keywords)


def is_honest_refusal(answer: str) -> bool:
    """
    برای سوالات out_of_scope: آیا مدل صادقانه گفته جواب را پیدا نکرده،
    به‌جای این‌که از دانش عمومی خودش (که می‌تواند برای این مدل خاص
    خودرو نادرست باشد) چیزی بسازد؟
    """
    refusal_phrases = ['پیدا نکردم', 'اطلاعاتی', 'موجود نیست', 'ذکر نشده', 'یافت نشد']
    return any(phrase in answer for phrase in refusal_phrases)


def run_evaluation(questions_path: str = '../eval_questions.json', verbose: bool = True) -> dict:
    with open(questions_path, encoding='utf-8') as f:
        questions = json.load(f)

    results = []

    for q in questions:
        result = answer_question(q['question'])
        source_pages = [p['page_range'] for p in result['sources']]

        if q['category'] == 'out_of_scope':
            retrieval_ok = None
            answer_ok = is_honest_refusal(result['answer'])
        else:
            retrieval_ok = page_in_expected(q['expected_page'], source_pages)
            answer_ok = keywords_in_answer(q.get('expected_keywords', []), result['answer'])

        record = {
            'id': q['id'],
            'question': q['question'],
            'category': q['category'],
            'answer': result['answer'],
            'source_pages': source_pages,
            'expected_page': q.get('expected_page'),
            'retrieval_ok': retrieval_ok,
            'answer_ok': answer_ok,
        }
        results.append(record)

        if verbose:
            status = '✓' if (answer_ok or retrieval_ok) else '✗'
            print(f"[{status}] #{q['id']} ({q['category']}): {q['question']}")
            print(f"      پاسخ: {result['answer'][:100]}")
            print(f"      صفحات منبع: {source_pages} | انتظار: {q.get('expected_page')}")
            print()

    # خلاصه‌ی آماری
    retrieval_results = [r['retrieval_ok'] for r in results if r['retrieval_ok'] is not None]
    answer_results = [r['answer_ok'] for r in results if r['answer_ok'] is not None]

    retrieval_accuracy = sum(retrieval_results) / len(retrieval_results) if retrieval_results else None
    answer_accuracy = sum(answer_results) / len(answer_results) if answer_results else None

    summary = {
        'total_questions': len(questions),
        'retrieval_accuracy': retrieval_accuracy,
        'answer_accuracy': answer_accuracy,
        'results': results,
    }

    if verbose:
        print('=' * 50)
        print(f"تعداد کل سوالات: {summary['total_questions']}")
        if retrieval_accuracy is not None:
            print(f"دقت بازیابی (Retrieval Accuracy): {retrieval_accuracy:.0%}")
        if answer_accuracy is not None:
            print(f"دقت پاسخ (Answer Accuracy): {answer_accuracy:.0%}")

    return summary


if __name__ == '__main__':
    summary = run_evaluation()
    with open('../eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print()
    print('نتایج کامل ذخیره شد: ../eval_results.json')

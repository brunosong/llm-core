import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context


load_dotenv(override=True)

MODEL = "gemini/gemini-3.1-flash-lite"
db_name = "vector_db"


class RetrievalEval(BaseModel):
    """검색(retrieval) 성능에 대한 평가 지표."""

    mrr: float = Field(description="평균 역순위(Mean Reciprocal Rank) - 모든 키워드에 대한 평균")
    ndcg: float = Field(description="정규화 할인 누적 이익(Normalized Discounted Cumulative Gain, 이진 관련성)")
    keywords_found: int = Field(description="상위 k개 결과에서 발견된 키워드 수")
    total_keywords: int = Field(description="찾아야 할 전체 키워드 수")
    keyword_coverage: float = Field(description="발견된 키워드의 비율(%)")


class AnswerEval(BaseModel):
    """LLM을 판사로 사용한(LLM-as-a-judge) 답변 품질 평가."""

    feedback: str = Field(
        description="참조 답변과 비교하고 검색된 컨텍스트를 바탕으로 평가한, 답변 품질에 대한 간결한 피드백"
    )
    accuracy: float = Field(
        description="참조 답변과 비교했을 때 사실적으로 얼마나 정확한가? 1(오답. 틀린 답은 반드시 1점) ~ 5(이상적 - 완벽하게 정확함). 허용 가능한 답변은 3점."
    )
    completeness: float = Field(
        description="질문의 모든 측면을 다루는 데 있어 답변이 얼마나 완전한가? 1(매우 부족 - 핵심 정보 누락) ~ 5(이상적 - 참조 답변의 모든 정보가 완전하게 제공됨). 참조 답변의 모든 정보가 포함된 경우에만 5점."
    )
    relevance: float = Field(
        description="답변이 질문에 얼마나 관련성이 있는가? 1(매우 부족 - 주제에서 벗어남) ~ 5(이상적 - 질문에 직접 답하고 불필요한 추가 정보 없음). 질문과 완전히 관련이 있고 불필요한 추가 정보가 없을 때만 5점."
    )


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """키워드 하나에 대한 역순위(reciprocal rank)를 계산 (대소문자 구분 안 함)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """할인 누적 이익(Discounted Cumulative Gain)을 계산."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # 순위는 1부터 시작하므로 i+2
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """키워드 하나에 대한 nDCG를 계산 (이진 관련성, 대소문자 구분 안 함)."""
    keyword_lower = keyword.lower()

    # 이진 관련성: 키워드를 찾으면 1, 아니면 0
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # 이상적인 DCG (최선의 경우: 키워드가 첫 번째 위치에 있는 경우)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    테스트 질문에 대한 검색 성능을 평가.

    Args:
        test: 질문과 키워드를 담고 있는 TestQuestion 객체
        k: 검색할 상위 문서 개수 (기본값 10)

    Returns:
        MRR, nDCG, 키워드 커버리지 지표를 담은 RetrievalEval 객체
    """
    # 공통 answer 모듈을 사용해 문서를 검색
    retrieved_docs = fetch_context(test.question)

    # MRR 계산 (모든 키워드에 대한 평균)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # nDCG 계산 (모든 키워드에 대한 평균)
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # 키워드 커버리지 계산
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    LLM을 판사로 사용해(LLM-as-a-judge) 답변 품질을 평가 (비동기).

    Args:
        test: 질문과 참조 답변을 담고 있는 TestQuestion 객체

    Returns:
        (AnswerEval 객체, 생성된 답변 문자열, 검색된 문서 리스트)의 튜플
    """
    # 공통 answer 모듈을 사용해 RAG 응답을 가져옴
    generated_answer, retrieved_docs = answer_question(test.question)

    # LLM 판사 프롬프트
    judge_messages = [
        {
            "role": "system",
            "content": "당신은 답변의 품질을 평가하는 전문 평가자입니다. 생성된 답변을 참조 답변과 비교하여 평가하세요. 완벽한 답변에만 5/5점을 주세요.",
        },
        {
            "role": "user",
            "content": f"""질문:
{test.question}

생성된 답변:
{generated_answer}

참조 답변:
{test.reference_answer}

다음 세 가지 기준으로 생성된 답변을 평가해 주세요:
1. 정확성(Accuracy): 참조 답변과 비교했을 때 사실적으로 얼마나 정확한가? 완벽한 답변에만 5/5점을 주세요.
2. 완전성(Completeness): 질문의 모든 측면을 다루면서 참조 답변의 모든 정보를 얼마나 충실히 담고 있는가?
3. 관련성(Relevance): 불필요한 추가 정보 없이 질문에 얼마나 직접적으로 답하는가?

각 기준에 대해 1(매우 부족)~5(이상적) 점수와 상세한 피드백을 제공하세요. 답이 틀렸다면 정확성 점수는 반드시 1점이어야 합니다.""",
        },
    ]

    # 구조화된 출력으로 LLM 판사를 호출 (비동기)
    judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)

    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval():
    """모든 검색 테스트를 평가."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """배치 비동기 실행으로 모든 테스트에 대한 답변을 평가."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int):
    """특정 테스트에 대한 평가를 실행 (CLI용 비동기 헬퍼)."""
    # 테스트 로드
    tests = load_tests("tests.jsonl")

    if test_number < 0 or test_number >= len(tests):
        print(f"오류: test_row_number는 0과 {len(tests) - 1} 사이여야 합니다")
        sys.exit(1)

    # 테스트 가져오기
    test = tests[test_number]

    # 테스트 정보 출력
    print(f"\n{'=' * 80}")
    print(f"테스트 #{test_number}")
    print(f"{'=' * 80}")
    print(f"질문: {test.question}")
    print(f"키워드: {test.keywords}")
    print(f"카테고리: {test.category}")
    print(f"참조 답변: {test.reference_answer}")

    # 검색 평가
    print(f"\n{'=' * 80}")
    print("검색 평가")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"키워드 발견: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"키워드 커버리지: {retrieval_result.keyword_coverage:.1f}%")

    # 답변 평가
    print(f"\n{'=' * 80}")
    print("답변 평가")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\n생성된 답변:\n{generated_answer}")
    print(f"\n피드백:\n{answer_result.feedback}")
    print("\n점수:")
    print(f"  정확성: {answer_result.accuracy:.2f}/5")
    print(f"  완전성: {answer_result.completeness:.2f}/5")
    print(f"  관련성: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """행 번호로 특정 테스트를 평가하는 CLI."""
    if len(sys.argv) != 2:
        print("사용법: uv run eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("오류: test_row_number는 정수여야 합니다")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()

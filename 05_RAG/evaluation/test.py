import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """예상 키워드와 참조 답변을 포함하는 테스트 질문."""

    question: str = Field(description="RAG 시스템에 물어볼 질문")
    keywords: list[str] = Field(description="검색된 컨텍스트에 반드시 포함되어야 하는 키워드")
    reference_answer: str = Field(description="이 질문에 대한 참조 답변")
    category: str = Field(description="질문 카테고리 (예: direct_fact, spanning, temporal)")


def load_tests() -> list[TestQuestion]:
    """JSONL 파일에서 테스트 질문을 로드."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    return tests

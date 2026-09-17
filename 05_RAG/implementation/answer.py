from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
import os
from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gemini-3.1-flash-lite"
google_api_key = os.getenv('GOOGLE_API_KEY')


# MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
당신은 Insurellm이라는 회사를 대표하는, 해박하고 친절한 어시스턴트입니다.
당신은 지금 사용자와 Insurellm에 대해 대화하고 있습니다.
관련이 있다면, 주어진 컨텍스트를 활용해 질문에 답하세요.
답을 모른다면 모른다고 말하세요.
컨텍스트:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# 이렇게 retriever 생성 시점에 k를 박아야 함
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# llm = ChatOpenAI(temperature=0, model_name=MODEL)

llm = ChatGoogleGenerativeAI(
    temperature=0, 
    model=MODEL,
    google_api_key=google_api_key
)


def fetch_context(question: str) -> list[Document]:
    """
    질문과 관련된 컨텍스트 문서를 검색합니다.
    """
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    사용자의 모든 메시지를 하나의 문자열로 합칩니다.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    """
    RAG를 사용해 주어진 질문에 답하고, 답변과 컨텍스트 문서를 함께 반환합니다.
    """
    history = history or []
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.text, docs

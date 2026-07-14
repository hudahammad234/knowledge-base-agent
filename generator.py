"""

Calls Gemini to generate the final answer from the grounded prompt.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL_NAME, GOOGLE_API_KEY
from prompt_builder import build_answer_prompt

_llm_instance = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,  # Temperature 0 is ideal for RAG to ensure deterministic & factual answers
        )
    return _llm_instance


def generate_answer(question: str, chunks: list, llm=None) -> str:
    """Generates a grounded answer for `question` using the retrieved `chunks`."""
    if llm is None:
        llm = get_llm()
    prompt = build_answer_prompt(question, chunks)
    response = llm.invoke(prompt)
    return response.content
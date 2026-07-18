"""
Calls Gemini to generate the final answer from the grounded prompt.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL_NAME, GEMINI_API_KEY
from prompt_builder import build_answer_prompt


_llm_instance = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

    return _llm_instance



def generate_answer(question: str, chunks: list, llm=None) -> str:
    """
    Generates a grounded answer for question using retrieved chunks.
    """

    if llm is None:
        llm = get_llm()


    from validator import validate_answer


    max_attempts = 2
    validation = None
    candidate_answer = ""


    for attempt in range(max_attempts):

        try:

            prompt = build_answer_prompt(
                question,
                chunks
            )


            response = llm.invoke(prompt)

            candidate_answer = response.content


            validation = validate_answer(
                question,
                candidate_answer,
                chunks
            )


            if (
                validation.is_supported
                and not validation.has_hallucination
                and validation.is_relevant
            ):


                if chunks:
                    avg_similarity = sum(
                        chunk.get("score", 0.0)
                        for chunk in chunks
                    ) / len(chunks)

                else:
                    avg_similarity = 0.0



                chunk_count = len(chunks)

                chunk_factor = min(
                    1.0,
                    (0.5 if chunk_count >= 1 else 0.0)
                    +
                    (0.25 if chunk_count >= 2 else 0.0)
                    +
                    (0.25 if chunk_count >= 3 else 0.0)
                )


                validator_confidence = (
                    validation.confidence
                    if validation.confidence is not None
                    else 1.0
                )


                final_confidence = (
                    avg_similarity * 0.40
                    +
                    chunk_factor * 0.20
                    +
                    validator_confidence * 0.40
                )


                confidence_percentage = round(
                    final_confidence * 100,
                    1
                )


                return (
                    f"{candidate_answer}\n\n"
                    f"[Confidence Score: {confidence_percentage}%]"
                )


            print(
                f"Attempt {attempt+1}/{max_attempts}: "
                "Validation failed"
            )


        except Exception as exc:

            print(
                f"Attempt {attempt+1}/{max_attempts} failed: {exc}"
            )


    return (
        "Failed to generate a valid answer after "
        f"{max_attempts} attempts."
    )

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
            temperature=0,  # Temperature 0 is ideal for RAG to ensure deterministic & factual answers
        )
    return _llm_instance


def generate_answer(question: str, chunks: list, llm=None) -> str:
    """Generates a grounded answer for `question` using the retrieved `chunks`."""
    if llm is None:
        llm = get_llm()
    from validator import validate_answer
    max_attempts=2

    for attempt in range(max_attempts):
      prompt = build_answer_prompt(question, chunks)
      response = llm.invoke(prompt)
      candidate_answer=response.content


      validation= validate_answer(question, candidate_answer, chunks)

      if validation.is_supported and not validation.has_hallucination and validation.is_relevant:
        
            if chunks:
                avg_similarity = sum(chunk.get('score', 0.0) for chunk in chunks) / len(chunks)
            else:
                avg_similarity = 0.0
                
            
            chunk_count = len(chunks)
            chunk_factor = min(1.0, (0.5 if chunk_count >= 1 else 0.0) + 
                                    (0.25 if chunk_count >= 2 else 0.0) + 
                                    (0.25 if chunk_count >= 3 else 0.0))
            
            
            validator_confidence = validation.confidence if validation.confidence is not None else 1.0
            
            final_confidence = (avg_similarity * 0.40) + (chunk_factor * 0.20) + (validator_confidence * 0.40)
            final_confidence_percentage = round(final_confidence * 100, 1)

          
            return f"{candidate_answer}\n\n[Confidence Score: {final_confidence_percentage}%]"
            

      print(f"Attempt {attempt+1}/{max_attempts}: Validation failed. Response: {candidate_answer}, Validation: {validation.dict()}")
      print("Retrying with a new answer...")


    return f"Failed to generate a valid answer after {max_attempts} attempts. Last validation result: {validation.dict()}"


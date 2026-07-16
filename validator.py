"""

Uses structured Gemini outputs to validate if generated answers are grounded
in the retrieved context and checks for hallucinations.
"""

from typing import Optional
from pydantic import BaseModel, Field
from generator import get_llm


class ValidationResult(BaseModel):
    is_supported: bool = Field(
        description="True if the generated answer is entirely supported by the provided context chunks. False if the answer states things not present in the context."
    )
    is_relevant: bool = Field(
        description="True if the generated answer is directly relevant to the user's question and addresses it. False if the answer is off-topic or fails to address the query."
    )
    has_hallucination: bool = Field(
        description="True if the generated answer contains factual claims or details that are completely absent from the context chunks (hallucinations)."
    )
    citations_correct: bool = Field(
        description="True if the citations, sources, or references mentioned in the answer are accurate and correctly correspond to the context chunks. False if they are incorrect, mismatched, or fabricated."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 regarding the alignment between context and answer."
    )
    verdict: str = Field(
        description="Brief text explaining why the answer is supported/not-supported or why a hallucination was detected."
    )


def validate_answer(question: str, answer: str, chunks: list) -> ValidationResult:
    """Validates the generated answer against the retrieved chunks using Gemini with Structured Outputs."""
    
    # If the assistant responded with the pre-defined out-of-scope answer, it is safe and validated
    out_of_scope_phrase = "I don't have enough information in the knowledge base to answer that."
    if out_of_scope_phrase in answer:
        return ValidationResult(
            is_supported=True,
            is_relevant=True,
            has_hallucination=False,
            citations_correct=True,
            confidence=1.0,
            verdict="Correctly identified that the knowledge base lacks enough information."
        )

    # Prepare context for the validator
    from prompt_builder import build_context_block
    context_block = build_context_block(chunks)

    validation_prompt = f"""You are an objective AI Quality Auditor. Your job is to compare a generated ANSWER against the retrieved CONTEXT chunks and determine if it is factually grounded.

Context Chunks:
{context_block}

User Question: {question}
Generated Answer: {answer}

Verify carefully:
1. Is every claim in the ANSWER fully supported by the CONTEXT?
2. Does the ANSWER contain any hallucinations (claims not mentioned in CONTEXT)?
"""

    llm = get_llm()
    # Use Structured Output with Pydantic schema to ensure deterministic validation results
    structured_llm = llm.with_structured_output(ValidationResult)
    
    try:
        result = structured_llm.invoke(validation_prompt)
        return result
    except Exception as e:
        # Fallback in case of API failure
        return ValidationResult(
            is_supported=False,
            is_relevant=False,
            has_hallucination=True,
            confidence=0.0,
            citations_correct=False,
            verdict=f"Validation failed to execute due to an error: {str(e)}"
        )
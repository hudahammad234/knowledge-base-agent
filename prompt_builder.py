"""

Builds the prompts sent to Gemini, including instructions for answer 
generation and query rewriting based on conversation history.
"""

from typing import List, Dict, Any

#prevent the hallucination and ensure the model only uses the retrieved sources to answer the question
ANSWER_SYSTEM_INSTRUCTIONS = """You are the AI Knowledge Assistant for a company. \
Answer the user's question using ONLY the provided context chunks below.

Strict rules:
- Never invent information that is not present in the context.
- If the context does not contain enough information to answer, respond exactly: \
"I don't have enough information in the knowledge base to answer that."
- Produce a professional, structured response (use short paragraphs or bullet points).
- After your answer, on a new line, list the sources you used in this exact format: \
Sources: <document_name> (p.<page_number>), <document_name> (p.<page_number>)
  - Omit the page part if page_number is not available (e.g. for TXT/CSV/MD/DOCX).
- Do not cite a source you did not actually use in the answer."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a numbered context block with citation tags."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        page_part = f", page {chunk['page_number']}" if chunk.get("page_number") else ""
        tag = f"[Chunk {i} | Source: {chunk['document_name']}{page_part} | chunk #{chunk['chunk_number']}]"
        blocks.append(f"{tag}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def build_answer_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Builds the complete prompt sent to Gemini for answer generation."""
    context_block = build_context_block(chunks)
    if not context_block:
        context_block = "(No relevant context was found in the knowledge base.)"

    return f"""{ANSWER_SYSTEM_INSTRUCTIONS}

Context:
{context_block}

Question: {question}

Answer:"""


def build_rewrite_prompt(question: str, history: str) -> str:
    """Builds a prompt to reformulate a follow-up or ambiguous query into a search-friendly query."""
    return f"""You are an expert search query rewriter. 
Your task is to take a user's question and the conversation history, and output a single, self-contained, and fully-specified search query that can be used to search a vector store database.

Rules:
- If the question is already specific and does not refer to history, output it exactly as is.
- If the question contains pronouns or implicit references (e.g., "What about maternity leave?", "Is it paid?", "Vacation"), resolve them using the conversation history.
- Do NOT output any conversational text, pleasantries, or explanations. Just output the rewritten search query.

Conversation History:
{history if history else "(No prior history)"}

User Question: {question}

Search Query:"""
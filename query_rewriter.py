"""
query_rewriter.py
Owner: Member 2 (Retrieval)

Rewrites the user's raw question into a complete, self-contained search query
BEFORE retrieval happens. This matters a lot for retrieval quality because:
  - Short/vague queries ("Vacation") embed poorly and match weak/irrelevant chunks.
  - Follow-up questions ("what about maternity leave?") have no meaning on their
    own without the previous turn's context.

Approach (documented for the assignment's "Document your approach" requirement):
  1. If there's no conversation history -> ask Gemini to expand the raw query
     into a clear, complete question using domain-neutral instructions.
  2. If there IS conversation history -> include the last N turns so Gemini can
     resolve references ("it", "that", "what about X") into a standalone query.
  3. We ask Gemini to return ONLY the rewritten query (no preamble), and we
     fall back to the original query if the API call fails, so retrieval never
     breaks because of the rewriting step.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import google.generativeai as genai

import config

logger = logging.getLogger(__name__)

genai.configure(api_key=config.GEMINI_API_KEY)

_REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for a document \
search system. Rewrite the user's question into a single, complete, \
self-contained search query that would retrieve the most relevant information \
from a document collection.

Rules:
- If conversation history is provided, resolve pronouns and references \
  (e.g. "it", "that policy", "what about X") using that history.
- Do not answer the question. Only rewrite it.
- Return ONLY the rewritten query, nothing else. No quotes, no explanation.
- If the original question is already clear and complete, return it unchanged.
"""


def rewrite_query(
    user_question: str,
    conversation_history: Optional[List[dict]] = None,
    max_history_turns: int = 3,
) -> str:
    """
    Rewrite a user question into a complete search query.

    Args:
        user_question: the raw question typed by the user.
        conversation_history: optional list of {"role": "user"/"assistant", "content": str}
            representing prior turns in the session (from memory.py).
        max_history_turns: how many recent turns to include as context.

    Returns:
        The rewritten query string. Falls back to `user_question` on any failure.
    """
    if not user_question or not user_question.strip():
        return user_question

    prompt = _build_prompt(user_question, conversation_history, max_history_turns)

    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        response = model.generate_content(
            [_REWRITE_SYSTEM_PROMPT, prompt],
            generation_config={"temperature": 0.2, "max_output_tokens": 100},
        )
        rewritten = (response.text or "").strip().strip('"')
        if not rewritten:
            raise ValueError("Empty rewrite response")

        logger.info("Query rewritten: %r -> %r", user_question, rewritten)
        return rewritten

    except Exception as exc:  # noqa: BLE001 - we never want rewriting to break retrieval
        logger.warning("Query rewrite failed (%s), falling back to original query", exc)
        return user_question


def _build_prompt(
    user_question: str,
    conversation_history: Optional[List[dict]],
    max_history_turns: int,
) -> str:
    if not conversation_history:
        return f'User question: "{user_question}"\n\nRewritten query:'

    recent = conversation_history[-max_history_turns:]
    history_text = "\n".join(f'{turn["role"]}: {turn["content"]}' for turn in recent)

    return (
        f"Conversation history:\n{history_text}\n\n"
        f'Current user question: "{user_question}"\n\n'
        f"Rewritten, self-contained query:"
    )

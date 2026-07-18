"""
query_rewriter.py
Owner: Member 2 (Retrieval)

Rewrites the user's raw question into a complete, self-contained search query
BEFORE retrieval happens.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from google import genai

import config


logger = logging.getLogger(__name__)


# New Gemini client
client = genai.Client(
    api_key=config.GEMINI_API_KEY
)


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

    if not user_question or not user_question.strip():
        return user_question


    prompt = _build_prompt(
        user_question,
        conversation_history,
        max_history_turns
    )


    try:

        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[
                _REWRITE_SYSTEM_PROMPT,
                prompt
            ],
        )


        rewritten = (
            response.text or ""
        ).strip().strip('"')


        if not rewritten:
            raise ValueError(
                "Empty rewrite response"
            )


        logger.info(
            "Query rewritten: %r -> %r",
            user_question,
            rewritten
        )


        return rewritten


    except Exception as exc:

        logger.warning(
            "Query rewrite failed (%s), falling back to original query",
            exc
        )

        return user_question



def _build_prompt(
    user_question: str,
    conversation_history: Optional[List[dict]],
    max_history_turns: int,
) -> str:


    if not conversation_history:

        return (
            f'User question: "{user_question}"\n\n'
            "Rewritten query:"
        )


    recent = conversation_history[-max_history_turns:]


    history_text = "\n".join(
        f'{turn["role"]}: {turn["content"]}'
        for turn in recent
    )


    return (
        f"Conversation history:\n{history_text}\n\n"
        f'Current user question: "{user_question}"\n\n'
        "Rewritten, self-contained query:"
    )



# --------------------------------------------------------------------------- #
# Integration helper
# --------------------------------------------------------------------------- #

def history_from_memory_turns(
    turns: List[dict]
) -> List[dict]:

    history: List[dict] = []


    for turn in turns:

        history.append(
            {
                "role": "user",
                "content": turn["question"]
            }
        )


        history.append(
            {
                "role": "assistant",
                "content": turn["answer"]
            }
        )


    return history

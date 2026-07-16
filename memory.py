'''
Manages conversation memory: stores turns per session so the assistant can
understand follow-up questions that reference previous turns.
'''

from typing import List, Dict
from datetime import datetime


class ConversationMemory:
    """In-memory store of conversation turns, keyed by session_id."""

    def __init__(self):
        self._sessions: Dict[str, List[dict]] = {}

    def get_history(self, session_id: str) -> List[dict]:
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, question: str, rewritten_query: str, answer: str, sources: List[str]) -> None:
        self._sessions.setdefault(session_id, []).append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
        })

    def get_history_text(self, session_id: str, last_n: int = 3) -> str:
        """Returns the last `last_n` turns formatted as plain text, suitable
        for feeding into the query rewriter prompt as conversation context."""
        turns = self.get_history(session_id)[-last_n:]
        if not turns:
            return ''
        lines = []
        for t in turns:
            lines.append(f"User asked: {t['question']}")
            lines.append(f"Assistant answered: {t['answer'][:300]}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
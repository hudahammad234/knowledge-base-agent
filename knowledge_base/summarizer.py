"""
knowledge_base/summarizer.py

BONUS FEATURE: Document Summarization during Indexing.

Generates a short, human-readable summary for each document the first time
it is indexed, and caches that summary on disk so it is never regenerated
on every restart -- only when a document is brand new, or its content has
actually changed (same file-hash philosophy used for incremental indexing
in retriever.py, kept as a separate, independent cache here).

This module does NOT import anything from retriever.py, generator.py, or
config.py -- the LLM is passed in by the caller, so this file stays
self-contained and independently testable, exactly like loader.py,
chunker.py, and metadata.py.
"""

import json
import os
from typing import Optional

# Long documents don't need to be sent in full just to get a short summary --
# truncating keeps summarization fast and cheap even for large PDFs.
MAX_SUMMARY_INPUT_CHARS = 8000

SUMMARY_PROMPT_TEMPLATE = """Summarize the following document in 3-5 concise sentences.
Focus on what the document covers (topics, policies, key facts) so that someone
skimming a knowledge base index can tell at a glance what's inside it, without
reading the full document. Do not add information that is not in the text.

Document: {document_name}

Content:
{content}

Summary:"""


def generate_document_summary(llm, document_name: str, full_text: str) -> str:
    """Calls the LLM once to produce a short summary of a document's full text.

    Args:
        llm: any LangChain chat model with an .invoke(prompt) -> response.content
             interface (e.g. the ChatGoogleGenerativeAI instance from generator.py).
        document_name: shown to the LLM for context (e.g. "HR_Policy.md").
        full_text: the document's raw text (all pages joined together).

    Returns:
        A short plain-text summary (3-5 sentences).
    """
    content = full_text.strip()
    if len(content) > MAX_SUMMARY_INPUT_CHARS:
        content = content[:MAX_SUMMARY_INPUT_CHARS] + "\n...(truncated)"

    prompt = SUMMARY_PROMPT_TEMPLATE.format(document_name=document_name, content=content)
    response = llm.invoke(prompt)
    return response.content.strip()


class DocumentSummaryStore:
    """A tiny JSON-backed cache: doc_id -> {"file_hash": ..., "summary": ...}.

    Kept separate from retriever.py's index_manifest.json on purpose: the
    summarization feature can be added, removed, or fail independently
    without ever touching the already-tested incremental-indexing logic.
    """

    def __init__(self, path: str):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get_cached(self, doc_id: str, current_hash: str) -> Optional[str]:
        """Returns the cached summary ONLY if the file hasn't changed since
        it was last summarized (hash still matches). Returns None if the
        document is new, or its content changed -- signalling the caller
        that a fresh summary must be generated."""
        entry = self._data.get(doc_id)
        if entry and entry.get("file_hash") == current_hash:
            return entry.get("summary")
        return None

    def set(self, doc_id: str, file_hash: str, summary: str):
        self._data[doc_id] = {"file_hash": file_hash, "summary": summary}
        self._save()

    def all_summaries(self) -> dict:
        """Returns {doc_id: summary} for every cached document -- useful for
        building a quick 'what's inside this knowledge base' overview."""
        return {doc_id: entry["summary"] for doc_id, entry in self._data.items()}


def summarize_if_needed(llm, store: DocumentSummaryStore, doc_id: str,
                         file_hash: str, document_name: str, full_text: str) -> str:
    """Convenience wrapper: returns the cached summary if the document is
    unchanged, otherwise generates a new one and caches it. This is the
    single function retriever.py needs to call during indexing."""
    cached = store.get_cached(doc_id, file_hash)
    if cached is not None:
        return cached

    summary = generate_document_summary(llm, document_name, full_text)
    store.set(doc_id, file_hash, summary)
    return summary

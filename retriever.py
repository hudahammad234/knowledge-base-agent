"""
retriever.py
Owner: Member 2 (Retrieval)

Responsibilities covered in this file:
  1. Vector Database connection (ChromaDB)
  2. Top-K semantic search
  3. Duplicate removal
  4. Intelligent context building (token-budget aware)

This module is intentionally decoupled from embeddings.py (Member 1's file).
It expects an `embed_fn(text: str) -> list[float]` callable to be passed in,
so whichever embedding model Member 1 finalizes, this file doesn't need to change.
Member 1's embeddings.py returns a LangChain Embeddings object instead of a
plain callable, so use `embed_fn_from_langchain()` below to adapt it -- see
that function's docstring for the one-line integration example.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import chromadb

import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with everything the generator/validator needs."""
    text: str
    document_name: str
    page_number: Optional[int]
    chunk_number: int
    score: float  # similarity score, higher = more relevant
    metadata: dict = field(default_factory=dict)

    def to_citation(self) -> str:
        page_part = f" (p.{self.page_number})" if self.page_number is not None else ""
        return f"{self.document_name}{page_part}"


# --------------------------------------------------------------------------- #
# Integration helpers (bridge Member 1's embeddings.py <-> this module)
# --------------------------------------------------------------------------- #

def embed_fn_from_langchain(embedding_model) -> Callable[[str], List[float]]:
    """
    Adapts a LangChain Embeddings object (what Member 1's
    embeddings.get_embedding_function() returns) into the plain
    `str -> list[float]` callable this module expects.

    Usage:
        from embeddings import get_embedding_function
        from retriever import embed_fn_from_langchain, VectorStore, retrieve

        embed_fn = embed_fn_from_langchain(get_embedding_function())
        store = VectorStore()
        chunks = retrieve("some question", embed_fn=embed_fn, vector_store=store)
    """
    return embedding_model.embed_query


def sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB rejects metadata values that are None (raises an error on
    `collection.add`). Member 1's ChunkMetadata sets page_number=None for
    file types with no native page concept (DOCX, TXT, MD, CSV), so any
    None values must be replaced before chunks are inserted into the
    vector store. Call this on `chunk_metadata.to_dict()` right before
    `collection.add(...)`.
    """
    sanitized = {}
    for key, value in metadata.items():
        sanitized[key] = "N/A" if value is None else value
    return sanitized


def _none_if_na(value):
    """Reverses sanitize_metadata()'s "N/A" sentinel back to None when
    reading chunks back out of the vector store, so citations render
    cleanly (e.g. no page number shown for a DOCX chunk)."""
    return None if value == "N/A" else value


# --------------------------------------------------------------------------- #
# Vector store wrapper
# --------------------------------------------------------------------------- #

class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # IMPORTANT: force cosine similarity explicitly. Chroma defaults to L2
        # distance, and the "1 - distance" similarity conversion below is only
        # correct for cosine distance (range 0-2). Without this, scores would
        # be meaningless once real (non-toy) embeddings are used.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        """Run a similarity search and return normalized RetrievedChunk objects."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        if not results["documents"] or not results["documents"][0]:
            return chunks

        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Chroma returns a distance (lower = more similar) by default (L2 or cosine
            # distance depending on config). Convert to a 0-1 "similarity-like" score.
            similarity = max(0.0, 1.0 - distance)
            chunks.append(
                RetrievedChunk(
                    text=text,
                    document_name=meta.get("document_name", "unknown"),
                    page_number=_none_if_na(meta.get("page_number")),
                    chunk_number=meta.get("chunk_number", -1),
                    score=round(similarity, 4),
                    metadata=meta,
                )
            )
        return chunks


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

def deduplicate_chunks(
    chunks: List[RetrievedChunk],
    similarity_threshold: float = config.DEDUP_SIMILARITY_THRESHOLD,
) -> List[RetrievedChunk]:
    """
    Remove near-duplicate chunks.

    Two layers of dedup:
      1. Exact text match (same chunk indexed twice, e.g. re-indexing bug).
      2. Near-duplicate text using a cheap character-overlap ratio (no extra
         embedding calls needed -- fast and good enough for chunk-level text).
    """
    seen_exact = set()
    unique: List[RetrievedChunk] = []

    for chunk in chunks:
        normalized = " ".join(chunk.text.split()).lower()

        if normalized in seen_exact:
            continue

        is_near_duplicate = False
        for existing in unique:
            existing_normalized = " ".join(existing.text.split()).lower()
            if _text_overlap_ratio(normalized, existing_normalized) >= similarity_threshold:
                is_near_duplicate = True
                break

        if is_near_duplicate:
            continue

        seen_exact.add(normalized)
        unique.append(chunk)

    return unique


def _text_overlap_ratio(a: str, b: str) -> float:
    """Cheap similarity ratio based on shared word sets (Jaccard)."""
    words_a, words_b = set(a.split()), set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #

def build_context(
    chunks: List[RetrievedChunk],
    max_tokens: int = config.MAX_CONTEXT_TOKENS,
) -> str:
    """
    Assemble retrieved chunks into a single context string for the prompt,
    respecting a rough token budget so we never send unnecessary information
    to Gemini. Chunks are already ranked by relevance (highest score first),
    so we greedily add chunks until the budget is exhausted.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    context_parts = []
    used_tokens = 0

    for chunk in sorted_chunks:
        chunk_tokens = _estimate_tokens(chunk.text)
        if used_tokens + chunk_tokens > max_tokens:
            logger.debug("Context token budget reached, stopping at %s chunks", len(context_parts))
            break

        source_label = f"[Source: {chunk.to_citation()} | Chunk #{chunk.chunk_number}]"
        context_parts.append(f"{source_label}\n{chunk.text.strip()}")
        used_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text)."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def retrieve(
    query: str,
    embed_fn: Callable[[str], List[float]],
    vector_store: VectorStore,
    top_k: int = config.DEFAULT_TOP_K,
) -> List[RetrievedChunk]:
    """
    Full retrieval pipeline: embed query -> search -> dedup.
    Returns a clean, deduplicated list of RetrievedChunk, ranked by relevance.
    """
    query_embedding = embed_fn(query)
    raw_chunks = vector_store.query(query_embedding, top_k=top_k)
    clean_chunks = deduplicate_chunks(raw_chunks)

    logger.info(
        "Retrieved %s chunks (%s after dedup) for query: %r",
        len(raw_chunks), len(clean_chunks), query,
    )
    return clean_chunks

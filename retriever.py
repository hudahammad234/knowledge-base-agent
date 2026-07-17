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

    # ------------------------------------------------------------------- #
    # Dict-style access compatibility layer
    # ------------------------------------------------------------------- #
    # Member 3's prompt_builder.py / validator.py treat each chunk as a plain
    # dict (chunk['text'], chunk.get('page_number')). Rather than asking
    # Member 3 to rewrite their code to use attribute access, RetrievedChunk
    # supports both styles: chunk.text AND chunk['text'] both work.
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


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


# --------------------------------------------------------------------------- #
# BONUS: Hybrid Search (Semantic + Keyword), fused with Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
#
# This section is 100% ADDITIVE. It does not modify retrieve(), VectorStore,
# or any existing function/signature above, so nothing that already calls
# retrieve() is affected. It's an opt-in alternative entry point.
#
# Why hybrid search: pure semantic (embedding) search sometimes misses exact
# keyword matches (e.g. an exact policy code, a person's name, a number) that
# don't embed distinctively. Keyword search (BM25) catches those, but misses
# paraphrases/synonyms that semantic search catches. Combining both gives
# better recall than either alone.
#
# Why Reciprocal Rank Fusion (RRF) instead of averaging raw scores: cosine
# similarity (0-1 range) and BM25 scores (unbounded, corpus-dependent range)
# are NOT on the same scale, so averaging them directly is meaningless
# without careful normalization. RRF sidesteps this entirely by only using
# each result's RANK (position) in its own list, not its raw score:
#
#     fused_score(chunk) = sum( 1 / (k + rank_in_list) )  for each list
#                           the chunk appears in
#
# A chunk that ranks highly in BOTH the semantic and keyword lists rises to
# the top of the fused list. k (default 60) is a standard damping constant
# from the original RRF paper (Cormack et al., 2009) that prevents very
# top-ranked results from dominating disproportionately.

def _fetch_full_corpus(vector_store: VectorStore):
    """Pulls every chunk currently in the collection (needed to build a BM25
    index, since BM25 scores a query against the WHOLE corpus, unlike vector
    search which Chroma indexes for fast approximate nearest-neighbor lookup)."""
    data = vector_store.collection.get(include=["documents", "metadatas"])
    return data["documents"], data["metadatas"]


def _keyword_search(
    query: str,
    documents: List[str],
    metadatas: List[dict],
    top_k: int,
) -> List[RetrievedChunk]:
    """BM25 keyword search over the full corpus. Requires `rank_bm25`
    (add to requirements: `rank_bm25`)."""
    from rank_bm25 import BM25Okapi  # lazy import: only needed if hybrid search is used

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue  # BM25 score of 0 means no keyword overlap at all -- not a real match
        meta = metadatas[idx]
        results.append(
            RetrievedChunk(
                text=documents[idx],
                document_name=meta.get("document_name", "unknown"),
                page_number=_none_if_na(meta.get("page_number")),
                chunk_number=meta.get("chunk_number", -1),
                score=float(scores[idx]),
                metadata=meta,
            )
        )
    return results


def hybrid_retrieve(
    query: str,
    embed_fn: Callable[[str], List[float]],
    vector_store: VectorStore,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_pool: int = 15,
    rrf_k: int = 60,
) -> List[RetrievedChunk]:
    """
    Hybrid retrieval: semantic (vector) search + keyword (BM25) search,
    merged with Reciprocal Rank Fusion, then deduplicated.

    Drop-in alternative to retrieve() -- same return type (List[RetrievedChunk]),
    so it works with build_context(), prompt_builder.py, and validator.py
    without any changes on their side.

    Args:
        candidate_pool: how many candidates to pull from EACH method before
            fusion (larger than top_k, so fusion has enough overlap to work with).
        rrf_k: RRF damping constant (60 is the standard default from the
            original paper; higher = flatter/less aggressive re-ranking).
    """
    if vector_store.count() == 0:
        return []

    # 1. Semantic candidates
    query_embedding = embed_fn(query)
    semantic_results = vector_store.query(query_embedding, top_k=candidate_pool)

    # 2. Keyword candidates
    documents, metadatas = _fetch_full_corpus(vector_store)
    keyword_results = _keyword_search(query, documents, metadatas, top_k=candidate_pool)

    # 3. Reciprocal Rank Fusion
    fused_scores: dict = {}
    chunk_lookup: dict = {}

    for rank, chunk in enumerate(semantic_results):
        key = f"{chunk.document_name}_{chunk.chunk_number}"
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_lookup[key] = chunk

    for rank, chunk in enumerate(keyword_results):
        key = f"{chunk.document_name}_{chunk.chunk_number}"
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_lookup.setdefault(key, chunk)

    ranked_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)[: top_k * 2]

    fused_chunks = []
    for key in ranked_keys:
        chunk = chunk_lookup[key]
        chunk.score = round(fused_scores[key], 4)  # replace with fused rank score
        fused_chunks.append(chunk)

    clean_chunks = deduplicate_chunks(fused_chunks)[:top_k]

    logger.info(
        "Hybrid retrieve: %s semantic + %s keyword candidates -> %s fused -> %s final for query: %r",
        len(semantic_results), len(keyword_results), len(fused_chunks), len(clean_chunks), query,
    )
    return clean_chunks

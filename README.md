# Retrieval Module — Member 2

This folder contains everything for the **Retrieval** part of the AI
Knowledge Assistant: vector database, semantic search, deduplication,
context building, and query rewriting.

## Files

| File | Purpose |
|---|---|
| `config.py` | Shared settings (Chroma path/collection name, top_k, token budget, Gemini model). Import from here instead of hardcoding values. |
| `retriever.py` | `VectorStore` (ChromaDB wrapper), `retrieve()` (embed → search → dedup), `deduplicate_chunks()`, `build_context()`. |
| `query_rewriter.py` | `rewrite_query()` — expands short/vague questions and resolves follow-up references using Gemini. |
| `docs_query_rewriting_approach.md` | Written explanation of the query rewriting approach (assignment requirement). |
| `requirements-retrieval.txt` | Python dependencies for this module. |

## Setup

```bash
pip install -r requirements-retrieval.txt
export GEMINI_API_KEY="your-key-here"   # required for query_rewriter.py
```

## Integrating with Member 1's `embeddings.py`

`retriever.py` never imports `embeddings.py` directly — it accepts an
`embed_fn(text: str) -> list[float]` callable as a parameter. Once Member 1's
embeddings module is ready:

```python
from embeddings import embed_text   # Member 1's function
from retriever import VectorStore, retrieve

store = VectorStore()
chunks = retrieve("your question", embed_fn=embed_text, vector_store=store)
```

No changes needed inside `retriever.py` or `query_rewriter.py`.

## Notes / things still to verify by Member 2

- [ ] Run `query_rewriter.py` end-to-end with a real `GEMINI_API_KEY` (not yet
      tested live — falls back gracefully to the original query if it fails,
      but hasn't been confirmed working against the real API).
- [ ] Test `retriever.py` against real indexed documents once Member 1
      finishes `embeddings.py` and `loader.py`/`chunker.py`.
- [ ] Confirm metadata keys (`document_name`, `page_number`, `chunk_number`)
      match exactly what Member 1 stores at indexing time — `retriever.py`
      expects those exact keys.

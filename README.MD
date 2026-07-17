AI Knowledge Assistant

RAG system for indexing documents (PDF, DOCX, TXT, MD, CSV) and answering questions using retrieved context. Features include query rewriting, deduplication, response validation, conversation memory, and evaluation.

Built for Shamsieh Technology Services — AI Track, Team Assignment 02.




Architecture



                     ┌─────────────────────────┐
                     │   knowledge_base_docs/    │  (PDF / DOCX / TXT / MD / CSV)
                     └────────────┬──────────────┘
                                  ▼
 Member 1   ┌─────────────────────────────────────────────┐
            │ knowledge_base/loader.py   -> discover + read │
            │ knowledge_base/metadata.py -> DocumentMetadata,│
            │                               ChunkMetadata,   │
            │                               file hashing      │
            │ knowledge_base/chunker.py  -> RecursiveChar-    │
            │                               TextSplitter      │
            │ knowledge_base/summarizer.py -> doc summariza-  │
            │                                 tion on index   │
            │ embeddings.py              -> HuggingFace/Gemini│
            └────────────────────┬────────────────────────┘
                                  ▼
 Member 2   ┌─────────────────────────────────────────────┐
            │ retriever.py                                  │
            │   VectorStore (Chroma, cosine)                │
            │   retrieve()          -> embed -> search ->    │
            │                           dedup                │
            │   hybrid_retrieve()   -> BONUS: BM25 + vector  │
            │                           fused with RRF        │
            │   build_context()     -> token-budget context   │
            │ query_rewriter.py     -> rewrite_query()         │
            │                          history_from_memory_    │
            │                          turns()                 │
            └────────────────────┬────────────────────────┘
                                  ▼
 Member 3   ┌─────────────────────────────────────────────┐
            │ prompt_builder.py -> build_answer_prompt()      │
            │ generator.py      -> generate_answer() (Gemini, │
            │                       validate -> retry loop)    │
            │ validator.py      -> validate_answer()           │
            │                       (is_supported, is_relevant,│
            │                       has_hallucination,          │
            │                       citations_correct)          │
            │ memory.py         -> ConversationMemory           │
            └────────────────────┬────────────────────────┘
                                  ▼
 Member 4   ┌─────────────────────────────────────────────┐
            │ main.py       -> wires the full pipeline        │
            │ evaluation.py -> 27 fixed Q/A pairs, scoring     │
            │ export.py     -> Markdown / TXT conversation      │
            │                  export                           │
            └─────────────────────────────────────────────┘



Project Structure



project/
├── knowledge_base/
│   ├── loader.py           # Member 1
│   ├── chunker.py          # Member 1
│   ├── metadata.py         # Member 1
│   └── summarizer.py       # Member 1 
├── embeddings.py           # Member 1
├── retriever.py            # Member 2
├── query_rewriter.py       # Member 2
├── prompt_builder.py       # Member 3
├── generator.py            # Member 3
├── validator.py            # Member 3
├── memory.py               # Member 3
├── export.py               # Member 4
├── evaluation.py           # Member 4
├── main.py                 # Member 4
├── requirements.txt        # Member 4
└── README.md               # Member 4



Setup

1.
Install dependencies: pip install -r requirements.txt

2.
Configure .env: Set GEMINI_API_KEY and optional EMBEDDING_PROVIDER, CHROMA_PERSIST_DIR, etc.

3.
Place documents: Add files to knowledge_base_docs/.

4.
Run: python main.py to load, chunk, embed, and evaluate documents.

Bonus Features Implemented

•
Document Summarization (knowledge_base/summarizer.py): Generates and caches a short summary per document on indexing.

•
Hybrid Search (Semantic + Keyword)(retriever.py): Combines semantic similarity search with keyword-based matching to retrieve more accurate and relevant documents.

•
confidence score(validator.py):Measures the level of certainty that the generated answer is accurate and supported by the retrieved information.

•
Automated Evaluation Report(evaluation.py): Generates a performance report automatically by analyzing retrieval accuracy, answer grounding, confidence scores, and overall evaluation results.
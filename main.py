"""
main.py
Owner: Member 4 (Evaluation, Export, Final Integration)

Wires together every other member's module into one runnable pipeline:
  1. Load + chunk the knowledge base documents        (Member 1)
  2. Embed the chunks and (re)build the vector store  (Member 1 + Member 2)
  3. Run one sample question through the full pipeline
  4. Run the fixed evaluation set and save a report    (Member 4)
"""

import uuid

import config
from knowledge_base.loader import discover_files, load_document
from knowledge_base.chunker import chunk_document
from embeddings import get_embedding_function
from retriever import VectorStore, embed_fn_from_langchain, sanitize_metadata
from memory import ConversationMemory
from evaluation import run_pipeline, run_evaluation, generate_evaluation_report

# NOTE: no other member's file defines where the source documents live.
# Adjust this to wherever your knowledge-base folder actually is.
KNOWLEDGE_BASE_DIR = "knowledge_base"


def validate_config():
    """Fails fast with a clear message if required settings are missing,
    instead of letting the pipeline crash deep inside query_rewriter/generator/validator."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file before running the assistant."
        )
    if config.EMBEDDING_PROVIDER == "gemini" and not config.GOOGLE_API_KEY:
        raise RuntimeError(
            "EMBEDDING_PROVIDER is 'gemini' but GOOGLE_API_KEY is not set."
        )


def load_all_documents(folder_path: str = KNOWLEDGE_BASE_DIR):
    """Discovers and loads every supported file in `folder_path`."""
    documents = []
    for file_path in discover_files(folder_path):
        documents.append(load_document(file_path))
    return documents


def chunk_documents(documents):
    """Chunks every loaded document into a single flat list of chunk dicts."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc["metadata"], doc["pages"]))
    return all_chunks


def rebuild_vector_store(chunks):
    """Embeds every chunk and upserts it into the Chroma collection.
    Uses upsert (not add) so re-running on an unchanged/updated knowledge base
    doesn't fail on duplicate ids -- this is what supports incremental
    re-indexing (Functional Requirement #1)."""
    embed_fn = embed_fn_from_langchain(get_embedding_function())
    store = VectorStore()

    if not chunks:
        return store

    ids = [c["metadata"].chunk_id for c in chunks]
    texts = [c["text"] for c in chunks]
    embeddings = [embed_fn(t) for t in texts]
    metadatas = [sanitize_metadata(c["metadata"].to_dict()) for c in chunks]

    store.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return store


if __name__ == "__main__":
    validate_config()

    documents = load_all_documents()
    chunks = chunk_documents(documents)
    rebuild_vector_store(chunks)

    memory = ConversationMemory()
    session_id = str(uuid.uuid4())[:8]

    result = run_pipeline("How many annual leave days do employees get?", session_id, memory)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Validation:", result["validation"])

    results = run_evaluation()
    report = generate_evaluation_report(results)
    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved evaluation_report.md")
    print(report[:1000])

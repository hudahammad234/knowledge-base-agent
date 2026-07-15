"""
Shared configuration for the AI Knowledge Assistant project.
All team members should import settings from here instead of hardcoding values.

NOTE: This file merges settings needed by Member 1 (embeddings.py / loader.py /
chunker.py) and Member 2 (retriever.py / query_rewriter.py) into ONE config.py,
since only one file with this name can exist in the project root.
"""

import os

# --------------------------------------------------------------------------- #
# Member 1 settings — Document Processing & Embeddings
# --------------------------------------------------------------------------- #

# "local" (default, free, offline HuggingFace model) or "gemini"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --------------------------------------------------------------------------- #
# Member 2 settings — Vector Database & Retrieval
# --------------------------------------------------------------------------- #

# --- Vector Database (ChromaDB) ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

# --- Retrieval ---
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.95"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))

# --- Gemini (used by query_rewriter.py only) ---
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

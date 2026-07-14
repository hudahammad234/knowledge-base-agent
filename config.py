"""
Shared configuration for the AI Knowledge Assistant project.
All team members should import settings from here instead of hardcoding values,
so the whole pipeline stays consistent (same collection name, same paths, etc).
"""

import os

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

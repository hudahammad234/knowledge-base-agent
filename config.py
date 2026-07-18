"""
Shared configuration for AI Knowledge Assistant
"""

import os
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Embeddings
# -------------------------

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "local"
)

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY",
    ""
)


# -------------------------
# ChromaDB
# -------------------------

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./chroma_store"
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "knowledge_base"
)


# -------------------------
# Retrieval
# -------------------------

DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "5")
)

DEDUP_SIMILARITY_THRESHOLD = float(
    os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.95")
)

MAX_CONTEXT_TOKENS = int(
    os.getenv("MAX_CONTEXT_TOKENS", "3000")
)


# -------------------------
# Gemini
# -------------------------

GEMINI_MODEL_NAME = os.getenv(
    "GEMINI_MODEL_NAME",
    "gemini-2.0-flash"
)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    GOOGLE_API_KEY
)

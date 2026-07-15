# AI Knowledge Assistant - Knowledge Base Agent

This repository contains the implementation of the **Knowledge Base Agent**, designed to process and prepare documents for semantic search and retrieval.

## 🚀 Implemented Features (Member 1)

I have implemented the core data ingestion and processing pipeline, which consists of the following modules inside the `knowledge_base` package:

### 1. Document Loading (`loader.py`)
- Supports loading and reading raw documents of different formats (such as PDF, MD, etc.).
- Extracts page-by-page text content efficiently while keeping track of metadata.

### 2. Metadata Extraction (`metadata.py`)
- Defines standard schemas (`DocumentMetadata` and `ChunkMetadata`) to organize document attributes.
- Tracks critical information like document ID, file name, file type, and page numbers.

### 3. Text Chunking (`chunker.py`)
- Splits raw text into manageable segments using `RecursiveCharacterTextSplitter`.
- Implements overlapping chunks to preserve contextual information across borders.
- Maintains sequential chunk numbers across multi-page documents to ensure accurate citations.

### 4. Embedding Generation (`embeddings.py`)
- Generates vector embeddings for text chunks using embedding models.
- Converts processed chunks into numerical vector representations.
- Prepares embeddings for semantic search and retrieval.

---

## 🛠️ Technology Stack
- **Python 3.10+**
- **LangChain Text Splitters** (for recursive chunking)
- **Git & GitHub** (for version control)

---

## 📂 Project Structure
```text
knowledge-base-agent/
│
├── knowledge_base/
│   ├── loader.py        # Document loading logic
│   ├── metadata.py      # Metadata schemas & models
│   └── chunker.py       # Text splitting and chunking logic
│
├── embeddings.py        # Embedding generation module
└── README.md            # Project documentation
```
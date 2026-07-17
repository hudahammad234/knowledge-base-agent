# AI Knowledge Assistant

A RAG-based system for indexing documents (PDF, DOCX, TXT, MD, CSV) and answering user questions using retrieved context. The system includes query rewriting, document deduplication, response validation, conversation memory, and evaluation capabilities.

Built for Shamsieh Technology Services — AI Track, Team Assignment 02.

## Architecture

The system consists of four main modules:

### Member 1 - Knowledge Base Processing
Responsible for document processing and indexing:
- `knowledge_base/loader.py`: Discovers and loads supported documents.
- `knowledge_base/metadata.py`: Handles document and chunk metadata with file hashing.
- `knowledge_base/chunker.py`: Splits documents using RecursiveCharacterTextSplitter.
- `knowledge_base/summarizer.py`: Generates document summaries during indexing.
- `embeddings.py`: Creates embeddings using HuggingFace/Gemini models.

### Member 2 - Retrieval System
Responsible for searching and preparing relevant context:
- `retriever.py`:
  - Uses VectorStore (Chroma with cosine similarity).
  - Retrieves relevant chunks and removes duplicates.
  - Supports hybrid retrieval using BM25 and vector search with RRF fusion.
  - Builds context within token limits.
- `query_rewriter.py`: Rewrites user queries using conversation history.

### Member 3 - Generation and Validation
Responsible for generating and validating responses:
- `prompt_builder.py`: Creates structured prompts for answer generation.
- `generator.py`: Generates answers using Gemini with validation and retry logic.
- `validator.py`: Checks answer support, relevance, hallucination, and citation correctness.
- `memory.py`: Manages conversation history.

### Member 4 - Pipeline, Evaluation, and Export
Responsible for connecting the system and evaluating performance:
- `main.py`: Connects and runs the complete pipeline.
- `evaluation.py`: Contains fixed Q/A evaluation pairs and calculates performance scores.
- `export.py`: Exports conversations into Markdown and TXT formats.

## Project Structure

project/
- knowledge_base/
  - loader.py
  - chunker.py
  - metadata.py
  - summarizer.py
- embeddings.py
- retriever.py
- query_rewriter.py
- prompt_builder.py
- generator.py
- validator.py
- memory.py
- export.py
- evaluation.py
- main.py
- requirements.txt
- README.md


Setup

1. Install dependencies: pip install -r requirements.txt

2. Configure .env: Set GEMINI_API_KEY and optional EMBEDDING_PROVIDER, CHROMA_PERSIST_DIR, etc.

3. Place documents: Add files to knowledge_base_docs/.

4. Run: python main.py to load, chunk, embed, and evaluate documents.

Bonus Features Implemented

•
Document Summarization (knowledge_base/summarizer.py): Generates and caches a short summary per document on indexing.

•
Hybrid Search (Semantic + Keyword)(retriever.py): Combines semantic similarity search with keyword-based matching to retrieve more accurate and relevant documents.

•
confidence score(validator.py):Measures the level of certainty that the generated answer is accurate and supported by the retrieved information.

•
Automated Evaluation Report(evaluation.py): Generates a performance report automatically by analyzing retrieval accuracy, answer grounding, confidence scores, and overall evaluation results.

# AI Knowledge Assistant

A RAG-based system for indexing documents (PDF, DOCX, TXT, MD, CSV) and answering user questions using retrieved context. The system includes query rewriting, document deduplication, response validation, conversation memory, and evaluation capabilities.

Built for Shamsieh Technology Services — AI Track, Team Assignment 02.

## Architecture

### Member 1 - Knowledge Base Processing

Responsible for document loading, processing, and indexing:

- `knowledge_base/loader.py`: Discovers and loads supported documents.
- `knowledge_base/metadata.py`: Handles document metadata, chunk metadata, and file hashing.
- `knowledge_base/chunker.py`: Splits documents using RecursiveCharacterTextSplitter.
- `knowledge_base/summarizer.py`: Generates and caches document summaries during indexing.
- `embeddings.py`: Creates embeddings using HuggingFace/Gemini models.

### Member 2 - Retrieval System

Responsible for retrieving relevant information:

- `retriever.py`:
  - Uses Chroma VectorStore with cosine similarity.
  - Retrieves relevant document chunks and removes duplicates.
  - Supports hybrid retrieval using BM25 and vector search combined with RRF fusion.
  - Builds context within token limits.
- `query_rewriter.py`: Rewrites user queries using conversation history.

### Member 3 - Generation and Validation

Responsible for generating and validating answers:

- `prompt_builder.py`: Builds structured prompts for answer generation.
- `generator.py`: Generates responses using Gemini with validation and retry logic.
- `validator.py`: Validates answers by checking support, relevance, hallucination, and citation correctness.
- `memory.py`: Manages conversation history.

### Member 4 - Pipeline, Evaluation, and Export

Responsible for connecting the complete system and evaluating performance:

- `main.py`: Runs and connects the full pipeline.
- `evaluation.py`: Contains evaluation questions and calculates performance scores.
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

## Setup

1. Install dependencies:


pip install -r requirements.txt


2. Configure `.env`:

Set `GEMINI_API_KEY` and optional settings such as `EMBEDDING_PROVIDER` and `CHROMA_PERSIST_DIR`.

3. Add documents:

Place supported documents inside `knowledge_base_docs/`.

4. Run the application:


python main.py


The system will load documents, split them into chunks, generate embeddings, build the vector store, and execute the pipeline.

## Bonus Features Implemented

- **Document Summarization (`knowledge_base/summarizer.py`)**:
Generates and caches a short summary for each document during the indexing process.

- **Hybrid Search (Semantic + Keyword) (`retriever.py`)**:
Combines semantic similarity search with keyword-based retrieval to improve document relevance and accuracy.

- **Confidence Score (`validator.py`)**:
Measures the certainty that the generated answer is accurate and supported by the retrieved information.

- **Automated Evaluation Report (`evaluation.py`)**:
Automatically generates a performance report by analyzing retrieval accuracy, answer grounding, confidence scores, and overall evaluation results.

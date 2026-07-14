# AI Knowledge Assistant (RAG Pipeline)

A robust Retrieval-Augmented Generation (RAG) assistant that indexes local company documents (such as employee handbooks, remote work policies, and FAQs) and provides context-grounded, validated answers using Google Gemini.

## Key Features Implemented

* **Session-Based Conversation Memory (`memory.py`)**: Maintains localized conversation state using an in-memory session store. It retains the last $N$ turns of dialogue, enabling the assistant to seamlessly resolve follow-up questions.
* **Query Rewriting (`query_rewriter.py`)**: Integrates Gemini to rewrite implicit or ambiguous follow-up questions (e.g., "What about maternity leave?") into fully-specified, independent search queries based on the dialogue history.
* **Structured Prompt Design (`prompt_builder.py`)**: Features strict, instruction-aligned templates designed to completely prevent hallucinations, enforcing the model to answer *only* using retrieved document chunks and cite sources properly.
* **Gemini Integration (`generator.py`)**: Streamlines model interaction utilizing `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) with optimal parameters (like temperature `0` for deterministic and factual output).
* **Response Validation & Hallucination Guard (`validator.py`)**: Uses structured Pydantic outputs with Gemini to audit the generated answers against retrieved source context, checking for factual grounding and reporting confidence scores.

---

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Copy `.env.example` to `.env` and fill in your Gemini API key:
    ```bash
    cp .env.example .env
    ```
    Inside your `.env` file:
    ```env
    GOOGLE_API_KEY=your_actual_gemini_api_key_here
    ```

---

## How to Run the Project

The application provides a CLI entry point via `main.py` with three main commands:

### 1. Document Indexing
To parse, chunk, and index the documents inside your knowledge base (`sample_kb/`):
```bash
python main.py index
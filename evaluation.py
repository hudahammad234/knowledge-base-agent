"""
evaluation.py
Owner: Member 4 (Evaluation, Export, Final Integration)

Runs the full RAG pipeline (rewrite -> retrieve -> generate -> validate)
against a fixed set of evaluation questions and produces a scored report.
"""

from typing import List, Dict, Any
from datetime import datetime

import config
from memory import ConversationMemory
from query_rewriter import rewrite_query, history_from_memory_turns
from retriever import VectorStore, embed_fn_from_langchain, retrieve
from embeddings import get_embedding_function
from generator import generate_answer
from validator import validate_answer


EVALUATION_QUESTIONS: List[Dict[str, str]] = [
    {"question": "How many annual leave days do employees get?", "expected_answer": "Employees receive 21 annual leave days per year."},
    {"question": "What is the maternity leave policy?", "expected_answer": "Female employees are entitled to 70 days of paid maternity leave."},
    {"question": "What about maternity leave?", "expected_answer": "Female employees are entitled to 70 days of paid maternity leave."},
    {"question": "Is sick leave paid?", "expected_answer": "Yes, employees get 14 paid sick leave days per year."},
    {"question": "Vacation", "expected_answer": "Employees receive 21 annual leave days per year."},
    {"question": "Can employees carry over unused leave to next year?", "expected_answer": "Up to 5 unused leave days may be carried over to the next year."},
    {"question": "What is the notice period for resignation?", "expected_answer": "Employees must give 30 days written notice before resignation."},
    {"question": "Are remote work days allowed?", "expected_answer": "Employees may work remotely up to 2 days per week with manager approval."},
    {"question": "What equipment does the company provide for remote work?", "expected_answer": "The company provides a laptop and a monthly internet allowance."},
    {"question": "Do I need approval to work from home?", "expected_answer": "Yes, remote work requires manager approval."},
    {"question": "What are the core working hours?", "expected_answer": "Core working hours are 10:00 AM to 3:00 PM, Sunday to Thursday."},
    {"question": "Is overtime compensated?", "expected_answer": "Overtime is compensated at 1.5x the hourly rate."},
    {"question": "What is the company's dress code?", "expected_answer": "Business casual dress code, with smart casual on Thursdays."},
    {"question": "What is expected regarding confidentiality?", "expected_answer": "Employees must keep company and client information confidential during and after employment."},
    {"question": "What happens if an employee violates the code of conduct?", "expected_answer": "Violations may result in disciplinary action up to and including termination."},
    {"question": "Is there a policy on workplace harassment?", "expected_answer": "The company has zero tolerance for harassment and provides a reporting channel."},
    {"question": "How are conflicts of interest handled?", "expected_answer": "Employees must disclose any conflict of interest to HR."},
    {"question": "What is the onboarding process for new employees?", "expected_answer": "New employees complete orientation, IT setup, and a 90-day probation period."},
    {"question": "How long is the probation period?", "expected_answer": "The probation period is 90 days."},
    {"question": "Who do new employees report to during onboarding?", "expected_answer": "New employees are assigned a direct manager and a buddy/mentor."},
    {"question": "What is the company's official name?", "expected_answer": "Dar Al-Watan Technologies (per sample FAQ)."},
    {"question": "How can employees contact HR?", "expected_answer": "Employees can contact HR via the internal HR portal or email."},
    {"question": "What benefits does the company offer?", "expected_answer": "Health insurance, annual leave, and a training budget."},
    {"question": "Is there a training budget for employees?", "expected_answer": "Yes, each employee has an annual training budget."},
    {"question": "What is the process for requesting equipment repairs?", "expected_answer": "Submit a ticket through the IT support portal."},
    {"question": "What is the company's stance on smoking indoors?", "expected_answer": "Smoking is not permitted inside company premises."},
    {"question": "What is the capital of France?", "expected_answer": "I don't have enough information in the knowledge base to answer that."},
]


# --------------------------------------------------------------------------- #
# Retrieval glue (bridges Member 1's embeddings.py + Member 2's retriever.py)
# --------------------------------------------------------------------------- #
# Lazily initialized so importing this module never triggers model loading /
# a Chroma connection unless retrieval is actually used.
_vector_store = None
_embed_fn = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embed_fn_from_langchain(get_embedding_function())
    return _embed_fn


def retrieve_relevant_chunks(query: str, top_k: int = config.DEFAULT_TOP_K):
    """Embeds `query` and returns the top-k deduplicated chunks from the vector store."""
    return retrieve(query, embed_fn=_get_embed_fn(), vector_store=_get_vector_store(), top_k=top_k)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(question: str, session_id: str, memory: ConversationMemory) -> Dict[str, Any]:
    history = history_from_memory_turns(memory.get_history(session_id))
    rewritten = rewrite_query(question, conversation_history=history)
    chunks = retrieve_relevant_chunks(rewritten)
    answer = generate_answer(question, chunks)
    validation = validate_answer(question, answer, chunks)

    sources = [
        f"{c['document_name']}" + (f" (p.{c['page_number']})" if c.get("page_number") else "")
        for c in chunks
    ]

    memory.add_turn(session_id, question, rewritten, answer, sources)

    return {
        "question": question,
        "rewritten_query": rewritten,
        "retrieved_chunks": chunks,
        "answer": answer,
        "validation": validation,
        "sources": sources,
    }


def run_evaluation(questions=None):
    questions = questions or EVALUATION_QUESTIONS
    memory = ConversationMemory()
    results = []

    for i, item in enumerate(questions):
        result = run_pipeline(item["question"], session_id=f"eval-{i}", memory=memory)
        validation = result["validation"]
        retrieval_correct = len(result["retrieved_chunks"]) > 0
        is_grounded = validation.is_supported and not validation.has_hallucination

        results.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "actual_answer": result["answer"],
            "retrieval_correct": retrieval_correct,
            "is_grounded": is_grounded,
            "confidence": validation.confidence,
            "verdict": validation.verdict,
            "overall_score": round((int(retrieval_correct) + int(is_grounded) + validation.confidence) / 3, 2),
        })

    return results


def generate_evaluation_report(results):
    lines = [
        "# Evaluation Report --- AI Knowledge Assistant",
        f"Generated: {datetime.now().isoformat()}",
        f"Total questions evaluated: {len(results)}",
        "",
    ]

    avg_score = sum(r["overall_score"] for r in results) / len(results) if results else 0
    grounded_rate = sum(r["is_grounded"] for r in results) / len(results) if results else 0
    retrieval_rate = sum(r["retrieval_correct"] for r in results) / len(results) if results else 0

    lines += [
        "## Summary",
        f"- Average overall score: **{avg_score:.2f}** / 1.00",
        f"- Grounded answer rate: **{grounded_rate:.0%}**",
        f"- Correct retrieval rate: **{retrieval_rate:.0%}**",
        "",
        "## Per-Question Results",
        "",
        "| # | Question | Retrieval OK? | Grounded? | Confidence | Score |",
        "|---|---|---|---|---|---|",
    ]

    for i, r in enumerate(results, start=1):
        lines.append(
            f"| {i} | {r['question'][:50]} | {'✅' if r['retrieval_correct'] else '❌'} | "
            f"{'✅' if r['is_grounded'] else '❌'} | {r['confidence']:.2f} | {r['overall_score']:.2f} |"
        )

    lines.append("")
    lines.append("## Full Details")
    for i, r in enumerate(results, start=1):
        lines += [
            f"### Q{i}: {r['question']}",
            f"**Expected:** {r['expected_answer']}",
            f"**Actual:** {r['actual_answer']}",
            f"**Retrieval correct:** {r['retrieval_correct']} | **Grounded:** {r['is_grounded']} | "
            f"**Confidence:** {r['confidence']:.2f} | **Verdict:** {r['verdict']}",
            "",
        ]

    return "\n".join(lines)

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
            "question": item["question"], "expected_answer": item["expected_answer"],
            "actual_answer": result["answer"], "retrieval_correct": retrieval_correct,
            "is_grounded": is_grounded, "confidence": validation.confidence,
            "verdict": validation.verdict,
            "overall_score": round((int(retrieval_correct) + int(is_grounded) + validation.confidence) / 3, 2),
        })
    return results


def generate_evaluation_report(results):
    lines = ["# Evaluation Report — AI Knowledge Assistant",
             f"Generated: {datetime.now().isoformat()}", f"Total questions evaluated: {len(results)}", ""]
    avg_score = sum(r["overall_score"] for r in results) / len(results) if results else 0
    grounded_rate = sum(r["is_grounded"] for r in results) / len(results) if results else 0
    retrieval_rate = sum(r["retrieval_correct"] for r in results) / len(results) if results else 0
    lines += ["## Summary", f"- Average overall score: **{avg_score:.2f}** / 1.00",
              f"- Grounded answer rate: **{grounded_rate:.0%}**",
              f"- Correct retrieval rate: **{retrieval_rate:.0%}**", "",
              "## Per-Question Results", "",
              "| # | Question | Retrieval OK? | Grounded? | Confidence | Score |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(results, start=1):
        lines.append(f"| {i} | {r['question'][:50]} | {'✅' if r['retrieval_correct'] else '❌'} | "
                      f"{'✅' if r['is_grounded'] else '❌'} | {r['confidence']:.2f} | {r['overall_score']:.2f} |")
    lines.append("")
    lines.append("## Full Details")
    for i, r in enumerate(results, start=1):
        lines += [f"### Q{i}: {r['question']}", f"**Expected:** {r['expected_answer']}",
                  f"**Actual:** {r['actual_answer']}",
                  f"**Retrieval correct:** {r['retrieval_correct']} | **Grounded:** {r['is_grounded']} | "
                  f"**Confidence:** {r['confidence']:.2f} | **Verdict:** {r['verdict']}", ""]
    return "\n".join(lines)

print("evaluation.py loaded")
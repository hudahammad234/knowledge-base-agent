def run_pipeline(question: str, session_id: str, memory: ConversationMemory) -> Dict[str, Any]:
    history_text = memory.get_history_text(session_id)
    rewritten = rewrite_query(question, history=history_text)
    chunks = retrieve_relevant_chunks(rewritten)
    answer = generate_answer(question, chunks)
    validation = validate_answer(question, answer, chunks)
    sources = [
        f"{c['document_name']}" + (f" (p.{c['page_number']})" if c.get("page_number") else "")
        for c in chunks
    ]
    memory.add_turn(session_id, question, rewritten, answer, sources)
    return {"question": question, "rewritten_query": rewritten, "retrieved_chunks": chunks,
            "answer": answer, "validation": validation, "sources": sources}
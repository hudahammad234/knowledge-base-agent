validate_config()
documents = load_all_documents()
chunks = chunk_documents(documents)
rebuild_vector_store(chunks)


memory = ConversationMemory()
session_id = str(uuid.uuid4())[:8]

result = run_pipeline("How many annual leave days do employees get?", session_id, memory)
print("Answer:", result["answer"])
print("Sources:", result["sources"])
print("Validation:", result["validation"])



results = run_evaluation()
report = generate_evaluation_report(results)
with open("evaluation_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("Saved evaluation_report.md")
print(report[:1000])
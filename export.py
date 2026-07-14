def _ensure_exports_dir():
    os.makedirs(EXPORTS_DIR, exist_ok=True)


def export_to_markdown(turns: List[dict], session_id: str) -> str:
    _ensure_exports_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORTS_DIR, f"conversation_{session_id}_{timestamp}.md")
    lines = [f"# Conversation Export — session `{session_id}`", ""]
    for i, t in enumerate(turns, start=1):
        lines.append(f"## Turn {i}")
        lines.append(f"**User:** {t['question']}")
        lines.append("")
        lines.append(f"**Assistant:** {t['answer']}")
        if t.get("sources"):
            lines.append("")
            lines.append(f"*Sources: {', '.join(t['sources'])}*")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def export_to_txt(turns: List[dict], session_id: str) -> str:
    _ensure_exports_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORTS_DIR, f"conversation_{session_id}_{timestamp}.txt")
    lines = [f"Conversation Export - session {session_id}", "=" * 50, ""]
    for i, t in enumerate(turns, start=1):
        lines.append(f"Turn {i}")
        lines.append(f"User: {t['question']}")
        lines.append(f"Assistant: {t['answer']}")
        if t.get("sources"):
            lines.append(f"Sources: {', '.join(t['sources'])}")
        lines.append("-" * 50)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

print("export.py loaded")
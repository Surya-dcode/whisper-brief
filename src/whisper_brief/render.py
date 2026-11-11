"""Markdown rendering for brief.md."""

from __future__ import annotations

from whisper_brief.models import Brief


def brief_to_markdown(brief: Brief) -> str:
    lines = [f"# {brief.title}", "", brief.summary, ""]

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {item}" for item in items)
        lines.append("")

    section("Key points", brief.key_points)
    section("Decisions", brief.decisions)
    section(
        "Action items",
        [f"{a.task} ({a.owner})" if a.owner else a.task for a in brief.action_items],
    )
    section("Open questions", brief.open_questions)
    lines.append(f"_Language: {brief.language}_")
    return "\n".join(lines) + "\n"

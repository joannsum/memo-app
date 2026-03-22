"""
vault.py — Save and load conversations as Markdown files.

Each session = one .md file with Obsidian-friendly frontmatter.
User profiles = vault/entities/<user>.md

This is the persistence layer. No database required.
"""

from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

VAULT_DIR = Path("vault")


def session_path(user_id: str) -> Path:
    d = VAULT_DIR / "sessions" / user_id
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return d / f"{ts}.md"


def write_header(path: Path, user_id: str):
    path.write_text(
        f"---\nuser: {user_id}\n"
        f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"tags: [memo, chat]\n---\n\n"
        f"# Session — {datetime.now().strftime('%B %d %Y, %H:%M')}\n\n",
        encoding="utf-8",
    )


def append_message(path: Path, role: str, content: str):
    label = "**You**" if role == "user" else "**Memo**"
    ts    = datetime.now().strftime("%H:%M")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"### {label} `{ts}`\n{content}\n\n")


def load_past_sessions(user_id: str, max_sessions: int = 3) -> str:
    d = VAULT_DIR / "sessions" / user_id
    if not d.exists():
        return ""
    files = sorted(d.glob("*.md"))
    if not files:
        return ""
    parts = []
    for p in files[-max_sessions:]:
        text = p.read_text(encoding="utf-8")
        # strip YAML frontmatter
        text = re.sub(r"^---.*?---\n", "", text, flags=re.DOTALL).strip()
        parts.append(text)
    return "\n\n---\n\n".join(parts)


def list_sessions(user_id: str) -> list[str]:
    d = VAULT_DIR / "sessions" / user_id
    if not d.exists():
        return []
    return sorted(f.name for f in d.glob("*.md"))


def write_entity_file(user_id: str, fact: str, tags: list[str], source_session: str = ""):
    """
    Write a single memory fact as an Obsidian-compatible .md file.

    Saved to vault/entities/<user_id>/<timestamp>.md with YAML frontmatter
    and [[wikilinks]] for cross-referencing in Obsidian's graph view.
    """
    d = VAULT_DIR / "entities" / user_id
    d.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{ts}_{_slugify(fact[:40])}.md"
    path = d / filename

    # Build YAML frontmatter
    tag_str = ", ".join(tags)
    lines = [
        "---",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        f"tags: [{tag_str}]",
        f"source_session: {source_session}",
        f"user: {user_id}",
        "---",
        "",
        f"# {fact}",
        "",
    ]

    # Add wikilinks for any capitalized proper nouns / project names
    # This makes the vault navigable in Obsidian's graph view.
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", fact)
    if words:
        links = " · ".join(f"[[{w}]]" for w in set(words))
        lines.append(f"Related: {links}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _slugify(text: str) -> str:
    """Turn a short string into a safe filename fragment."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text[:40]

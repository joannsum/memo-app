"""
memory.py — Pure Python memory management. No LangChain.

Three memory types taught in this workshop:
  1. BufferMemory     — keep every message (simple, expensive)
  2. WindowMemory     — keep last N messages (cheap, forgets)
  3. SummaryMemory    — summarize old messages (best of both)

"""

from __future__ import annotations
import httpx
from dataclasses import dataclass, field
from typing import Literal

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


class BufferMemory:
    """
    Keeps every message in full.
    Simple but token count grows forever.
    """
    name = "Buffer Memory"

    def __init__(self):
        self.messages: list[Message] = []

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def build_history(self) -> str:
        return "\n".join(
            f"{'Human' if m.role == 'user' else 'Memo'}: {m.content}"
            for m in self.messages
        )

    def token_estimate(self) -> int:
        return len(self.build_history()) // 4

    def to_dict(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class WindowMemory:
    """
    Keeps only the last K messages.
    Cheap but loses older context entirely.
    """
    name = "Window Memory"

    def __init__(self, k: int = 6):
        self.k = k
        self.messages: list[Message] = []

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def build_history(self) -> str:
        window = self.messages[-self.k:]
        return "\n".join(
            f"{'Human' if m.role == 'user' else 'Memo'}: {m.content}"
            for m in window
        )

    def token_estimate(self) -> int:
        return len(self.build_history()) // 4

    def to_dict(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages[-self.k:]]


class SummaryMemory:
    """
    Keeps recent messages verbatim.
    Summarizes older ones when buffer exceeds max_tokens.
    Best of both worlds — this is what production apps use.
    """
    name = "Summary Memory"

    def __init__(self, max_tokens: int = 400):
        self.max_tokens   = max_tokens
        self.messages:  list[Message] = []
        self.summary:   str = ""   # running summary of old messages

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        # Compress if we're over budget
        if self._current_tokens() > self.max_tokens:
            self._compress()

    def _current_tokens(self) -> int:
        return len(self.build_history()) // 4

    def _compress(self):
        """Summarize the oldest half of messages, keep the rest verbatim."""
        half = len(self.messages) // 2
        to_summarize = self.messages[:half]
        self.messages = self.messages[half:]

        conversation = "\n".join(
            f"{'Human' if m.role == 'user' else 'Memo'}: {m.content}"
            for m in to_summarize
        )

        prompt = (
            "Summarize this conversation in 2-3 sentences, "
            "focusing on key facts about the human and important topics discussed. "
            "Be concise.\n\n"
            f"{conversation}\n\nSummary:"
        )

        try:
            resp = httpx.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            new_summary = resp.json().get("response", "").strip()
            self.summary = (self.summary + " " + new_summary).strip() if self.summary else new_summary
        except Exception:
            # If summarization fails, just drop old messages silently
            pass

    def build_history(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Earlier summary: {self.summary}]")
        parts += [
            f"{'Human' if m.role == 'user' else 'Memo'}: {m.content}"
            for m in self.messages
        ]
        return "\n".join(parts)

    def token_estimate(self) -> int:
        return len(self.build_history()) // 4

    def to_dict(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

"""
embeddings.py — Generate text embeddings using Ollama's local embedding API.

Uses the nomic-embed-text model which runs entirely on your machine.
Setup:  ollama pull nomic-embed-text

This module is intentionally simple — one async function that takes
a string and returns a list of floats. Students can read it in under
a minute and understand exactly what an embedding is.
"""

from __future__ import annotations
import httpx

# Ollama exposes an embeddings endpoint alongside the generate endpoint.
# nomic-embed-text produces 768-dimensional vectors — good quality, fast, local.
EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


async def embed(text: str) -> list[float]:
    """
    Convert a piece of text into a numeric vector (embedding).

    The embedding captures the *meaning* of the text — similar ideas
    produce vectors that are close together in 768-dimensional space.
    ChromaDB uses this to find memories that are semantically relevant
    to the user's current message.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
        )
    data = resp.json()
    return data["embedding"]

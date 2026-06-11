"""Document chunking: split long texts into retrieval-sized pieces.

Embedding a whole document into one vector dilutes its content; retrieval
works best over passages of a few hundred tokens. The chunker splits on
paragraph boundaries first, sentences second, and packs pieces greedily up
to ``max_chars`` with ``overlap`` characters of trailing context carried
into the next chunk so statements spanning a boundary stay findable.
"""

from __future__ import annotations

import re

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str, max_chars: int) -> list[str]:
    """Paragraphs, then sentences for paragraphs that are still too long."""
    units: list[str] = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _SENTENCE_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_chars:
                units.append(sent)
            else:  # pathological run-on: hard wrap
                units.extend(
                    sent[i : i + max_chars] for i in range(0, len(sent), max_chars)
                )
    return units


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Consecutive chunks share up to ``overlap`` characters of context (the
    tail of the previous chunk, cut on a word boundary). The overlap is
    clamped to a third of ``max_chars`` so small chunk sizes stay valid
    with the default overlap.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    overlap = min(overlap, max_chars // 3)
    if len(text.strip()) <= max_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    units = _split_units(text, max_chars)
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        tail = ""
        if overlap > 0 and chunks:
            tail = chunks[-1][-overlap:]
            cut = tail.find(" ")
            if cut != -1:
                tail = tail[cut + 1 :]
        current = f"{tail}\n\n{unit}" if tail and len(tail) + len(unit) + 2 <= max_chars else unit
    if current:
        chunks.append(current)
    return chunks

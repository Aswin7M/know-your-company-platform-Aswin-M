"""Paragraph-aware chunker that keeps citation metadata on every chunk.

Sizes are *approximate tokens* (chars / 3.5), which tracks WordPiece tokenisers
closely enough for English prose. The default 500 tokens stays under the 512-token
limit of the default embedding model so chunks are not silently truncated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from models.source import Evidence

CHARS_PER_TOKEN = 3.5
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    company: str
    company_slug: str
    source_id: str
    url: str
    title: str
    source_type: str
    chunk_index: int


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _hard_split(text: str, max_chars: int) -> list[str]:
    out, words, current = [], text.split(), ""
    for w in words:
        if len(w) > max_chars:                       # pathological token (e.g. base64)
            if current:
                out.append(current)
                current = ""
            out.extend(w[i:i + max_chars] for i in range(0, len(w), max_chars))
            continue
        if current and len(current) + 1 + len(w) > max_chars:
            out.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        out.append(current)
    return out


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    out, current = [], ""
    for sentence in _SENTENCE_RE.split(paragraph):
        if len(sentence) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.extend(_hard_split(sentence, max_chars))
        elif current and len(current) + 1 + len(sentence) > max_chars:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        out.append(current)
    return out


def _tail(text: str, n_chars: int) -> str:
    """Last ~n_chars of text, starting on a word boundary."""
    if n_chars <= 0 or len(text) <= n_chars:
        return "" if n_chars <= 0 else text
    piece = text[-n_chars:]
    space = piece.find(" ")
    return piece[space + 1:] if space != -1 else piece


def split_text(text: str, chunk_size: int = 500, overlap: int = 60) -> list[str]:
    """Split text into chunks of ~chunk_size tokens with ~overlap tokens of carry-over."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))
    max_chars = max(50, int(chunk_size * CHARS_PER_TOKEN))
    overlap_chars = int(overlap * CHARS_PER_TOKEN)

    units: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        units.extend([line] if len(line) <= max_chars else _split_long(line, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_chars:
            chunks.append(current)
            tail = _tail(current, overlap_chars)
            current = f"{tail}\n{unit}" if tail else unit
            if len(current) > max_chars:             # overlap would overflow -> drop it
                current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_evidence(evidence: Evidence, company: str, company_slug: str,
                   chunk_size: int = 500, overlap: int = 60) -> list[Chunk]:
    return [
        Chunk(
            text=piece, company=company, company_slug=company_slug,
            source_id=evidence.source_id, url=evidence.url, title=evidence.title,
            source_type=evidence.source_type, chunk_index=i,
        )
        for i, piece in enumerate(split_text(evidence.content, chunk_size, overlap))
    ]


def chunk_all(evidence_items: Iterable[Evidence], company: str, company_slug: str,
              chunk_size: int = 500, overlap: int = 60) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ev in evidence_items:
        chunks.extend(chunk_evidence(ev, company, company_slug, chunk_size, overlap))
    return chunks

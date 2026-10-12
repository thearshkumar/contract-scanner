"""Python port of web/src/core/chunk.ts. Must stay identical: the app scans chunks produced by the TS version and
the model is trained and evaluated on chunks from this one. tests/test_chunk.py checks both against a golden file."""

from __future__ import annotations

from dataclasses import dataclass

BREAKS = ["\n\n", "\n", ". ", "; ", ", ", " "]


@dataclass(frozen=True)
class Chunk:
    id: int
    start: int
    end: int
    text: str


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 400) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _break_before(text, start, end)
        body = text[start:end]
        if body.strip():
            chunks.append(Chunk(len(chunks), start, end, body))
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt)
        if space != -1 and space < end:
            nxt = space + 1
        start = nxt
    return chunks


def _break_before(text: str, start: int, end: int) -> int:
    floor = start + (end - start) // 2
    for sep in BREAKS:
        # Same as JS text.lastIndexOf(sep, end - sep.length): last match lying entirely before `end`.
        i = text.rfind(sep, 0, end)
        if i >= floor:
            return i + len(sep)
    return end

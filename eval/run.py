"""Turns per-chunk model outputs into per-contract spans and scores them (span + presence metrics)."""

from __future__ import annotations

from dataclasses import dataclass

from eval.chunk import Chunk, chunk_text
from eval.cuad import Contract, Span
from eval.metrics import aggregate, match_spans, presence
from eval.prompt import CLAUSE_TYPES, gold_in_chunk, parse, target, to_spans


@dataclass
class Example:
    title: str
    chunk: Chunk
    gold: list[Span]

    @property
    def target(self) -> str:
        return target(self.gold)


def examples(contracts: list[Contract]) -> list[Example]:
    return [Example(c.title, ch, gold_in_chunk(c.spans, ch)) for c in contracts for ch in chunk_text(c.text)]


def dedupe(spans: list[Span]) -> list[Span]:
    """Overlapping chunks can report the same passage twice: keep one per type when they overlap by >= 50%."""
    kept: list[Span] = []
    for s in sorted(spans, key=lambda s: (s.clause_type, s.start, -(s.end - s.start))):
        dup = False
        for k in kept:
            if k.clause_type != s.clause_type:
                continue
            inter = min(k.end, s.end) - max(k.start, s.start)
            if inter > 0 and inter >= 0.5 * min(k.end - k.start, s.end - s.start):
                dup = True
                break
        if not dup:
            kept.append(s)
    return sorted(kept, key=lambda s: (s.start, s.clause_type))


def score(contracts: list[Contract], outputs: dict[tuple[str, int], str]) -> dict:
    span_rows, presence_rows = [], []
    parse_fail = misses = items_total = 0
    for c in contracts:
        pred: list[Span] = []
        for ch in chunk_text(c.text):
            items, ok = parse(outputs.get((c.title, ch.id), ""))
            parse_fail += not ok
            items_total += len(items)
            spans, m = to_spans(ch, items)
            misses += m
            pred.extend(spans)
        pred = dedupe(pred)
        span_rows.append(match_spans(c.text, c.spans, pred))
        presence_rows.append(presence(c.clause_types, {s.clause_type for s in pred}))
    n_chunks = sum(len(chunk_text(c.text)) for c in contracts)
    return {
        "contracts": len(contracts),
        "chunks": n_chunks,
        "parse_failure_rate": round(parse_fail / n_chunks, 4) if n_chunks else 0.0,
        "items": items_total,
        "anchor_miss_rate": round(misses / items_total, 4) if items_total else 0.0,
        "span": aggregate(span_rows, CLAUSE_TYPES),
        "presence": aggregate(presence_rows, CLAUSE_TYPES),
    }

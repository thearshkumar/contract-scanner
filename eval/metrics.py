"""Scoring for clause extraction.

Span level: a predicted span matches a gold span of the same clause type when their token IoU >= 0.5;
matching is one-to-one (greedy by IoU). Reported per clause type and micro-averaged.
Document level: did the model say the clause type is present at all (presence P/R/F1).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from eval.cuad import Span

IOU_THRESHOLD = 0.5
_TOKEN = re.compile(r"\S+")


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, other: PRF) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4)}


def token_spans(text: str, start: int, end: int) -> set[int]:
    """Indices of whitespace tokens of `text` that overlap [start, end)."""
    return {i for i, m in enumerate(_TOKEN.finditer(text)) if m.start() < end and m.end() > start}


def iou(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def match_spans(text: str, gold: list[Span], pred: list[Span]) -> dict[str, PRF]:
    """Per clause type TP/FP/FN for one contract."""
    out: dict[str, PRF] = defaultdict(PRF)
    for ctype in {s.clause_type for s in gold} | {s.clause_type for s in pred}:
        g = [token_spans(text, s.start, s.end) for s in gold if s.clause_type == ctype]
        p = [token_spans(text, s.start, s.end) for s in pred if s.clause_type == ctype]
        pairs = sorted(((iou(pi, gi), i, j) for i, pi in enumerate(p) for j, gi in enumerate(g)), reverse=True)
        used_p, used_g = set(), set()
        for score, i, j in pairs:
            if score < IOU_THRESHOLD:
                break
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
        out[ctype] = PRF(tp=len(used_p), fp=len(p) - len(used_p), fn=len(g) - len(used_g))
    return out


def presence(gold: set[str], pred: set[str]) -> dict[str, PRF]:
    out: dict[str, PRF] = defaultdict(PRF)
    for ctype in gold | pred:
        out[ctype] = PRF(tp=int(ctype in gold and ctype in pred), fp=int(ctype in pred and ctype not in gold),
                         fn=int(ctype in gold and ctype not in pred))
    return out


def aggregate(per_contract: list[dict[str, PRF]], all_types: list[str]) -> dict:
    by_type = {t: PRF() for t in all_types}
    micro = PRF()
    for d in per_contract:
        for t, prf in d.items():
            by_type.setdefault(t, PRF()).add(prf)
            micro.add(prf)
    macro_f1 = sum(p.f1 for p in by_type.values()) / len(by_type) if by_type else 0.0
    return {"micro": micro.as_dict(), "macro_f1": round(macro_f1, 4), "by_type": {t: p.as_dict() for t, p in by_type.items()}}

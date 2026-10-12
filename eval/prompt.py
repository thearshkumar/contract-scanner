"""The one prompt/output contract shared by training, offline eval and (ported to TS) the live app.

The model gets one chunk and returns a JSON array of clauses. Instead of copying whole clauses (CUAD spans run to
thousands of characters), each item gives verbatim *anchors*: the first and last ANCHOR_WORDS words. `locate()` maps
anchors back to exact character offsets, so highlights stay exact while output tokens stay small.
"""

from __future__ import annotations

import json
import re

from eval.chunk import Chunk
from eval.cuad import Span

ANCHOR_WORDS = 10

# CUAD v1's 41 categories (names as in CUAD_v1.json question ids).
CLAUSE_TYPES = [
    "Affiliate License-Licensee", "Affiliate License-Licensor", "Agreement Date", "Anti-Assignment", "Audit Rights",
    "Cap On Liability", "Change Of Control", "Competitive Restriction Exception", "Covenant Not To Sue",
    "Document Name", "Effective Date", "Exclusivity", "Expiration Date", "Governing Law", "Insurance",
    "Ip Ownership Assignment", "Irrevocable Or Perpetual License", "Joint Ip Ownership", "License Grant",
    "Liquidated Damages", "Minimum Commitment", "Most Favored Nation", "No-Solicit Of Customers",
    "No-Solicit Of Employees", "Non-Compete", "Non-Disparagement", "Non-Transferable License",
    "Notice Period To Terminate Renewal", "Parties", "Post-Termination Services", "Price Restrictions",
    "Renewal Term", "Revenue/Profit Sharing", "Rofr/Rofo/Rofn", "Source Code Escrow", "Termination For Convenience",
    "Third Party Beneficiary", "Uncapped Liability", "Unlimited/All-You-Can-Eat-License", "Volume Restriction",
    "Warranty Duration",
]
_TYPE_LOOKUP = {t.lower(): t for t in CLAUSE_TYPES}

SYSTEM = (
    "You review commercial contracts. Find every passage in the excerpt that belongs to one of these clause types:\n"
    + "\n".join(f"- {t}" for t in CLAUSE_TYPES)
    + "\n\nReply with ONLY a JSON array, one object per passage, in the order they appear:\n"
    '[{"type": "<clause type from the list>", "begin": "<first words of the passage>", '
    '"end": "<last words of the passage>"}]\n'
    f"`begin` and `end` must be copied word for word from the excerpt, at most {ANCHOR_WORDS} words each. "
    "For a short passage, `begin` and `end` may be the same. If nothing matches, reply []."
)


def messages(chunk_text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Excerpt:\n<<<\n{chunk_text}\n>>>"}]


# ---------- targets (training) ----------

def _words(text: str) -> list[re.Match]:
    return list(re.finditer(r"\S+", text))


def anchors(span_text: str) -> tuple[str, str]:
    """Verbatim first/last ANCHOR_WORDS words of a passage (original whitespace kept)."""
    w = _words(span_text)
    if not w:
        return span_text, span_text
    if len(w) <= ANCHOR_WORDS:
        return span_text[w[0].start() : w[-1].end()], span_text[w[0].start() : w[-1].end()]
    begin = span_text[w[0].start() : w[ANCHOR_WORDS - 1].end()]
    end = span_text[w[-ANCHOR_WORDS].start() : w[-1].end()]
    return begin, end


def gold_in_chunk(spans: list[Span], chunk: Chunk) -> list[Span]:
    """Gold spans lying entirely inside the chunk (spans cut by a boundary are seen whole in the overlap)."""
    return [s for s in spans if s.start >= chunk.start and s.end <= chunk.end]


def target(spans: list[Span]) -> str:
    items = []
    for s in sorted(spans, key=lambda s: (s.start, s.end, s.clause_type)):
        b, e = anchors(s.text)
        items.append({"type": s.clause_type, "begin": " ".join(b.split()), "end": " ".join(e.split())})
    return json.dumps(items, ensure_ascii=False)


# ---------- parsing model output ----------

_OBJECT = re.compile(r"\{[^{}]*\}")


def parse(output: str) -> tuple[list[dict], bool]:
    """Returns (items, ok). Tolerates prose or code fences around the array; drops items with unknown types.

    If the array is malformed or cut off (output hit max_tokens), every complete `{...}` object is still salvaged,
    but `ok` is False so the failure is counted.
    """
    text = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
    start, end = text.find("["), text.rfind("]")
    raw, ok = None, False
    if start != -1 and end > start:
        try:
            raw = json.loads(text[start : end + 1])
            ok = isinstance(raw, list)
        except json.JSONDecodeError:
            raw = None
    if not ok:
        if start == -1 and text.strip() == "":
            return [], True
        raw = []
        for m in _OBJECT.finditer(text[start:] if start != -1 else text):
            try:
                raw.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
    items = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        ctype = _TYPE_LOOKUP.get(str(it.get("type", "")).strip().lower())
        begin = str(it.get("begin") or it.get("quote") or "").strip()
        end_ = str(it.get("end") or begin).strip()
        if ctype and begin:
            items.append({"type": ctype, "begin": begin, "end": end_})
    return items, ok


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapses whitespace runs to one space; returns the new text and a map new index -> original index."""
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
        idx.append(i)
    return "".join(out), idx


def locate(text: str, begin: str, end: str) -> tuple[int, int] | None:
    """Character offsets in `text` of the passage from `begin` through `end` (whitespace-insensitive)."""
    norm, idx = _normalise(text)
    nb, ne = " ".join(begin.split()), " ".join(end.split())
    if not nb:
        return None
    b = norm.find(nb)
    if b == -1:
        b = norm.lower().find(nb.lower())
        if b == -1:
            return None
    e = norm.find(ne, b) if ne else -1
    if e == -1 and ne:
        e = norm.lower().find(ne.lower(), b)
    stop = (e + len(ne)) if e != -1 else (b + len(nb))
    return idx[b], idx[stop - 1] + 1


def to_spans(chunk: Chunk, items: list[dict]) -> tuple[list[Span], int]:
    """Absolute spans for parsed items; also returns how many items could not be located (citation misses)."""
    spans, misses = [], 0
    for it in items:
        loc = locate(chunk.text, it["begin"], it["end"])
        if loc is None:
            misses += 1
            continue
        s, e = chunk.start + loc[0], chunk.start + loc[1]
        spans.append(Span(it["type"], s, e, chunk.text[loc[0] : loc[1]]))
    return spans, misses

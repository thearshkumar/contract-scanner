"""Builds the CUAD example-clause library for "Similar in CUAD" (Vectorize index `cs-clauses`).

    uv run python -m eval.library            # writes data/library.ndjson (gitignored)

Train split only, so the demo contracts (test split) never retrieve their own clauses. At most PER_TYPE clauses per
type: 75 x 41 x 1024 dims = 3.15M stored dimensions, which leaves room under the 5M free allowance that this account
shares with search.arshkumar.com (https://developers.cloudflare.com/vectorize/platform/pricing/).
Selection is deterministic (hash order) and spread across contracts: one clause per contract per type first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval import cuad

PER_TYPE = 75
MIN_CHARS, MAX_CHARS = 40, 1200
OUT = Path("data/library.ndjson")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build(contracts: list[cuad.Contract]) -> list[dict]:
    by_type: dict[str, list[tuple[str, str, cuad.Span]]] = {}
    for c in contracts:
        for s in c.spans:
            text = " ".join(s.text.split())
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                by_type.setdefault(s.clause_type, []).append((c.title, text, s))
    rows = []
    for ctype in sorted(by_type):
        cands = sorted(by_type[ctype], key=lambda t: _h(f"{t[0]}:{t[2].start}"))
        picked, seen_titles, seen_text = [], set(), set()
        for pass_ in (0, 1):  # first pass: one per contract, for variety
            for title, text, s in cands:
                if len(picked) >= PER_TYPE:
                    break
                if text in seen_text or (pass_ == 0 and title in seen_titles):
                    continue
                picked.append((title, text, s))
                seen_titles.add(title)
                seen_text.add(text)
        for title, text, s in picked:
            rows.append({"id": _h(f"{ctype}:{title}:{s.start}")[:32], "clauseType": ctype, "text": text,
                         "source": title})
    return rows


def main() -> None:
    train, _ = cuad.split(cuad.load())
    rows = build(train)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["clauseType"]] = counts.get(r["clauseType"], 0) + 1
    print(json.dumps({"rows": len(rows), "types": len(counts), "min_per_type": min(counts.values()),
                      "stored_dims": len(rows) * 1024}))


if __name__ == "__main__":
    main()

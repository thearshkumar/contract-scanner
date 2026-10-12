"""Precomputed demo scans for the site: three CUAD test contracts with the fine-tuned model's real outputs.

    modal volume get cs-adapters runs/r8-v1/outputs.jsonl out/r8-v1/outputs.jsonl
    uv run python -m eval.demos               # writes web/public/demo/*.json

The spans are exactly what the live pipeline would produce (same chunker, parser, anchor locator and dedupe), from
the r8-v1 held-out eval run, so a demo costs no GPU time. CUAD is CC BY 4.0 (The Atticus Project); the attribution
ships inside each file and on the page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from eval import cuad
from eval.chunk import chunk_text
from eval.prompt import parse, to_spans
from eval.run import dedupe

OUTPUTS = Path("out/r8-v1/outputs.jsonl")
DEST = Path("web/public/demo")
MIN_CHARS, MAX_CHARS = 12_000, 40_000
WATCH = {"Uncapped Liability", "Non-Compete", "Exclusivity", "Liquidated Damages", "Change Of Control",
         "Anti-Assignment", "Termination For Convenience", "Most Favored Nation", "Ip Ownership Assignment",
         "Minimum Commitment", "Rofr/Rofo/Rofn", "Revenue/Profit Sharing", "Covenant Not To Sue"}
ATTRIBUTION = ("From CUAD v1 (Hendrycks et al., NeurIPS 2021), The Atticus Project, CC BY 4.0. "
               "A public SEC filing; spans are this project's model output, not CUAD's labels.")


def kind(title: str) -> str:
    """'ACME_2019-EX-10.1-SUPPLY AGREEMENT' and '..._Ex-10.19_123_Ex-10.19_Content License Agreement' -> the kind."""
    return re.split(r"[-_]EX-\d+(?:\.\d+)*[-_]", title, flags=re.IGNORECASE)[-1].strip().title()


def main() -> None:
    outputs: dict[tuple[str, int], str] = {}
    for line in OUTPUTS.read_text().splitlines():
        r = json.loads(line)
        outputs[(r["title"], r["chunk"])] = r["output"]
    _, test = cuad.split(cuad.load())

    scored = []
    for c in test:
        if not MIN_CHARS <= len(c.text) <= MAX_CHARS:
            continue
        spans = []
        for ch in chunk_text(c.text):
            items, _ = parse(outputs.get((c.title, ch.id), ""))
            spans.extend(to_spans(ch, items)[0])
        spans = dedupe(spans)
        watch = {s.clause_type for s in spans} & WATCH
        scored.append((len(watch), len({s.clause_type for s in spans}), c, spans))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2].title))

    picked, kinds = [], set()
    for w, n, c, spans in scored:
        if kind(c.title) in kinds:
            continue
        picked.append((c, spans))
        kinds.add(kind(c.title))
        if len(picked) == 3:
            break

    DEST.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (c, spans) in enumerate(picked, start=1):
        slug = f"demo-{i}"
        doc = {"id": slug, "title": kind(c.title), "source": c.title, "attribution": ATTRIBUTION, "text": c.text,
               "spans": [{"type": s.clause_type, "start": s.start, "end": s.end} for s in spans]}
        (DEST / f"{slug}.json").write_text(json.dumps(doc, ensure_ascii=False))
        index.append({"id": slug, "title": doc["title"], "chars": len(c.text), "clauses": len(spans)})
        print(slug, doc["title"], len(c.text), "chars,", len(spans), "spans")
    (DEST / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Pins the prompt/parse contract shared with the live app (web/src/core/scan-prompt.ts reads the same fixture).

Regenerate after an intentional change with:  UPDATE_GOLDEN=1 uv run pytest eval/tests/test_prompt_golden.py
"""

import json
import os
from pathlib import Path

from eval.cuad import Span
from eval.prompt import locate as locate_
from eval.prompt import messages, parse
from eval.run import dedupe

FIX = Path(__file__).parent / "fixtures" / "prompt_golden.json"

CHUNK = "Section 1.  This Agreement shall\nautomatically renew for successive one (1) year terms.\n\n" \
        "Section 2. This Agreement is governed by the laws of the State of Delaware."

PARSE_INPUTS = [
    '[{"type": "Governing Law", "begin": "This Agreement is governed", "end": "State of Delaware."}]',
    '```json\n[{"type": "renewal term", "begin": "This Agreement shall automatically", "end": "year terms."}]\n```',
    '<think>\n\n</think>\n\n[]',
    "",
    "Nothing found.",
    '[{"type": "Parties", "begin": "Acme Inc."}, {"type": "Not A Type", "begin": "x"}, {"type": "Parties"}]',
    '[{"type": "Governing Law", "begin": "governed by", "end": "Delaware."}, {"type": "Renewal Term", "beg',
    '[{"type": "Parties", "quote": "Acme Corp"}, 5, "str", {"type": "Agreement Date", "begin": "", "end": "x"}]',
]

LOCATE_INPUTS = [
    ["This Agreement shall automatically renew", "year terms."],
    ["this agreement is GOVERNED", "state of delaware."],
    ["Section 2.", ""],
    ["not in the text", "x"],
    ["", "x"],
    ["Section 1. This", "missing end"],
]

DEDUPE_INPUT = [
    ["Parties", 0, 50], ["Parties", 10, 40], ["Parties", 45, 120], ["Governing Law", 0, 50],
    ["Renewal Term", 200, 260], ["Renewal Term", 230, 300], ["Renewal Term", 100, 101],
]


def build() -> dict:
    return {
        "chunk": CHUNK,
        "messages": messages(CHUNK),
        "parse": [{"input": s, "items": parse(s)[0], "ok": parse(s)[1]} for s in PARSE_INPUTS],
        "locate": [{"begin": b, "end": e, "result": list(r) if (r := locate_(CHUNK, b, e)) else None}
                   for b, e in LOCATE_INPUTS],
        "dedupe": {
            "input": DEDUPE_INPUT,
            "output": [[s.clause_type, s.start, s.end]
                       for s in dedupe([Span(t, a, b, "") for t, a, b in DEDUPE_INPUT])],
        },
    }


def test_matches_golden():
    got = build()
    if os.environ.get("UPDATE_GOLDEN"):
        FIX.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n")
    assert got == json.loads(FIX.read_text())

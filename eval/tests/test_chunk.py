import json
from itertools import pairwise
from pathlib import Path

import pytest

from eval.chunk import chunk_text

FIX = Path(__file__).parent / "fixtures"
TEXT = (FIX / "chunk_input.txt").read_text()


@pytest.mark.parametrize("case", json.loads((FIX / "chunk_golden.json").read_text()))
def test_matches_golden(case):
    got = [[c.start, c.end] for c in chunk_text(TEXT, case["maxChars"], case["overlap"])]
    assert got == case["chunks"]


def test_invariants():
    chunks = chunk_text(TEXT, 1000, 100)
    assert chunks[0].start == 0 and chunks[-1].end == len(TEXT)
    for c in chunks:
        assert TEXT[c.start : c.end] == c.text and len(c.text) <= 1000
    for a, b in pairwise(chunks):
        assert b.start < a.end  # overlap, no gaps


def test_edge_cases():
    assert chunk_text("") == []
    assert len(chunk_text("x" * 2500, 1000, 100)) == 3
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)

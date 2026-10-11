"""CUAD v1 loader and a deterministic contract-level train/test split.

Source: Hugging Face `theatticusproject/cuad` (CC BY 4.0, The Atticus Project), file `CUAD_v1/CUAD_v1.json`
(SQuAD format: one entry per contract, one question per clause type). The official train/test files live in
CUAD's GitHub `data.zip`; this project builds its own split instead (sha256 of the contract title, 80/20),
so it is reproducible from the HF copy alone. Scores are therefore not directly comparable to the CUAD paper.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

HF_REPO = "theatticusproject/cuad"
HF_REVISION = "a3c393f5d103fd0c516374e4fdff676c8176dcb1"  # pinned
HF_FILE = "CUAD_v1/CUAD_v1.json"
DEFAULT_CACHE = Path("data/cuad")


@dataclass(frozen=True)
class Span:
    clause_type: str
    start: int
    end: int  # exclusive
    text: str


@dataclass
class Contract:
    title: str
    text: str
    spans: list[Span] = field(default_factory=list)

    @property
    def clause_types(self) -> set[str]:
        return {s.clause_type for s in self.spans}


def download(cache: Path = DEFAULT_CACHE) -> Path:
    """Fetches CUAD_v1.json once (about 40 MB) and returns its local path."""
    path = cache / "CUAD_v1.json"
    if path.exists():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE.replace(' ', '%20')}"
    log.info("downloading CUAD", extra={"url": url})
    urllib.request.urlretrieve(url, path)
    return path


def parse(raw: dict) -> list[Contract]:
    contracts = []
    for entry in raw["data"]:
        for para in entry["paragraphs"]:
            text = para["context"]
            spans = []
            for qa in para["qas"]:
                clause_type = qa["id"].split("__", 1)[1] if "__" in qa["id"] else qa["question"]
                for ans in qa.get("answers", []):
                    start = ans["answer_start"]
                    span_text = ans["text"]
                    if text[start : start + len(span_text)] != span_text:  # rare offset drift: re-anchor
                        found = text.find(span_text)
                        if found == -1:
                            log.warning("dropped unanchored span", extra={"title": entry["title"], "type": clause_type})
                            continue
                        start = found
                    spans.append(Span(clause_type, start, start + len(span_text), span_text))
            contracts.append(Contract(entry["title"], text, sorted(set(spans), key=lambda s: (s.start, s.end, s.clause_type))))
    return contracts


def load(cache: Path = DEFAULT_CACHE) -> list[Contract]:
    with open(download(cache)) as f:
        return parse(json.load(f))


def split_of(title: str, test_fraction: float = 0.2) -> str:
    """Deterministic: a contract is in `test` iff its title hash falls in the bottom `test_fraction`."""
    h = int(hashlib.sha256(title.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "train"


def split(contracts: list[Contract], test_fraction: float = 0.2) -> tuple[list[Contract], list[Contract]]:
    train = [c for c in contracts if split_of(c.title, test_fraction) == "train"]
    test = [c for c in contracts if split_of(c.title, test_fraction) == "test"]
    return train, test


def clause_types(contracts: list[Contract]) -> list[str]:
    return sorted({s.clause_type for c in contracts for s in c.spans})

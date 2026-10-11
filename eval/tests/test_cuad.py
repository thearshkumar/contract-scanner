from eval.cuad import Span, clause_types, parse, split, split_of

TEXT = "This Agreement may be terminated by either party. The term renews automatically each year."
RAW = {
    "data": [
        {
            "title": "ACME_Supply",
            "paragraphs": [
                {
                    "context": TEXT,
                    "qas": [
                        {"id": "ACME_Supply__Termination For Convenience", "question": "q",
                         "answers": [{"text": "may be terminated by either party", "answer_start": 15}]},
                        # offset drift (+3) must be re-anchored
                        {"id": "ACME_Supply__Renewal Term", "question": "q",
                         "answers": [{"text": "renews automatically", "answer_start": 63}]},
                        {"id": "ACME_Supply__Non-Compete", "question": "q", "answers": []},
                        {"id": "ACME_Supply__Governing Law", "question": "q",
                         "answers": [{"text": "not in the text", "answer_start": 0}]},
                    ],
                }
            ],
        }
    ]
}


def test_parse_spans_and_offsets():
    [c] = parse(RAW)
    assert c.title == "ACME_Supply"
    assert c.clause_types == {"Termination For Convenience", "Renewal Term"}  # empty + unanchored dropped
    for s in c.spans:
        assert TEXT[s.start : s.end] == s.text


def test_split_is_deterministic_and_disjoint():
    titles = [f"Contract_{i}" for i in range(500)]
    assert [split_of(t) for t in titles] == [split_of(t) for t in titles]
    test_share = sum(split_of(t) == "test" for t in titles) / len(titles)
    assert 0.12 < test_share < 0.28


def test_split_partitions_contracts():
    from eval.cuad import Contract

    cs = [Contract(f"C{i}", "x") for i in range(50)]
    train, test = split(cs)
    assert len(train) + len(test) == 50
    assert not {c.title for c in train} & {c.title for c in test}


def test_clause_types_sorted_unique():
    [c] = parse(RAW)
    assert clause_types([c, c]) == ["Renewal Term", "Termination For Convenience"]
    assert isinstance(c.spans[0], Span)

import json

from eval.chunk import Chunk
from eval.cuad import Contract, Span
from eval.prompt import CLAUSE_TYPES, anchors, locate, messages, parse, target, to_spans
from eval.run import dedupe, score

TEXT = "1. Term.  This Agreement shall\nautomatically renew for successive one (1) year terms unless either party " \
       "gives written notice.\n\n2. Law. This Agreement is governed by the laws of Delaware."


def span(ctype, phrase):
    s = TEXT.index(phrase)
    return Span(ctype, s, s + len(phrase), phrase)


RENEW = span("Renewal Term", "This Agreement shall\nautomatically renew for successive one (1) year terms unless either "
                             "party gives written notice.")
LAW = span("Governing Law", "This Agreement is governed by the laws of Delaware.")


def test_41_types_in_prompt():
    assert len(CLAUSE_TYPES) == 41 and len(set(CLAUSE_TYPES)) == 41
    assert all(t in messages("x")[0]["content"] for t in CLAUSE_TYPES)


def test_anchors_are_verbatim_and_bounded():
    b, e = anchors(RENEW.text)
    assert b in RENEW.text and e in RENEW.text
    assert len(b.split()) == 10 and len(e.split()) == 10
    short = anchors("Delaware law applies.")
    assert short == ("Delaware law applies.", "Delaware law applies.")


def test_target_roundtrip_recovers_exact_spans():
    chunk = Chunk(0, 0, len(TEXT), TEXT)
    items, ok = parse(target([LAW, RENEW]))
    assert ok and [i["type"] for i in items] == ["Renewal Term", "Governing Law"]
    spans, misses = to_spans(chunk, items)
    assert misses == 0
    assert {(s.start, s.end) for s in spans} == {(RENEW.start, RENEW.end), (LAW.start, LAW.end)}


def test_locate_is_whitespace_and_case_tolerant():
    loc = locate(TEXT, "this agreement shall automatically renew", "gives written notice.")
    assert loc == (RENEW.start, RENEW.end)
    assert locate(TEXT, "not in the text", "x") is None


def test_parse_tolerates_wrappers_and_rejects_garbage():
    out = '<think>\nhmm\n</think>\nHere you go:\n```json\n[{"type": "governing law", "begin": "This Agreement is governed", "end": "laws of Delaware."}]\n```'
    items, ok = parse(out)
    assert ok and items == [{"type": "Governing Law", "begin": "This Agreement is governed", "end": "laws of Delaware."}]
    assert parse("[]") == ([], True)
    assert parse("") == ([], True)
    assert parse("no clauses here")[1] is False
    assert parse('[{"type": "Made Up", "begin": "x"}]') == ([], True)
    assert parse("[{broken")[1] is False


def test_parse_salvages_complete_objects_from_truncated_output():
    cut = '[{"type": "Parties", "begin": "Acme Inc", "end": "Acme Inc"}, {"type": "Governing Law", "begin": "This Agr'
    items, ok = parse(cut)
    assert ok is False
    assert items == [{"type": "Parties", "begin": "Acme Inc", "end": "Acme Inc"}]


def test_chunk_offsets_are_absolute():
    chunk = Chunk(3, 100, 100 + len(TEXT), TEXT)
    spans, _ = to_spans(chunk, [{"type": "Governing Law", "begin": "This Agreement is governed", "end": "Delaware."}])
    assert spans[0].start == 100 + LAW.start


def test_dedupe_across_overlapping_chunks():
    a = Span("Governing Law", 10, 60, "")
    b = Span("Governing Law", 12, 60, "")
    c = Span("Renewal Term", 10, 60, "")
    assert dedupe([a, b, c]) == [a, c]


def test_score_perfect_and_empty():
    contract = Contract("C1", TEXT, [RENEW, LAW])
    from eval.chunk import chunk_text

    [ch] = chunk_text(TEXT)
    perfect = score([contract], {("C1", ch.id): target([RENEW, LAW])})
    assert perfect["span"]["micro"]["f1"] == 1.0 and perfect["presence"]["micro"]["f1"] == 1.0
    empty = score([contract], {("C1", ch.id): "[]"})
    assert empty["span"]["micro"]["recall"] == 0.0 and empty["parse_failure_rate"] == 0.0
    garbage = score([contract], {("C1", ch.id): "I think there is a renewal clause"})
    assert garbage["parse_failure_rate"] == 1.0
    assert json.dumps(perfect)  # serialisable

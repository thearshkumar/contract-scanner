import pytest

from eval.cuad import Span
from eval.metrics import PRF, aggregate, iou, match_spans, presence, token_spans

TEXT = "Either party may terminate this Agreement on thirty days notice. Liability is capped at fees paid."


def span(ctype, phrase, text=TEXT):
    s = text.index(phrase)
    return Span(ctype, s, s + len(phrase), phrase)


def test_token_iou():
    a = token_spans(TEXT, 0, 26)  # "Either party may terminate"
    assert len(a) == 4
    assert iou(a, a) == 1.0
    assert iou(set(), set()) == 0.0


def test_exact_and_partial_matches():
    gold = [span("Termination", "Either party may terminate this Agreement on thirty days notice.")]
    pred_good = [span("Termination", "may terminate this Agreement on thirty days notice.")]  # 8/10 tokens
    pred_bad = [span("Termination", "Either party")]  # 2/10 tokens
    assert match_spans(TEXT, gold, pred_good)["Termination"].as_dict()["f1"] == 1.0
    r = match_spans(TEXT, gold, pred_bad)["Termination"]
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_wrong_type_is_fp_and_fn():
    gold = [span("Cap On Liability", "Liability is capped at fees paid.")]
    pred = [span("Termination", "Liability is capped at fees paid.")]
    r = match_spans(TEXT, gold, pred)
    assert (r["Cap On Liability"].fn, r["Termination"].fp) == (1, 1)


def test_one_to_one_matching():
    gold = [span("Termination", "Either party may terminate")]
    pred = [span("Termination", "Either party may terminate")] * 2
    r = match_spans(TEXT, gold, pred)["Termination"]
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)


def test_presence_and_aggregate():
    per = [presence({"A", "B"}, {"A", "C"}), presence({"A"}, {"A"})]
    agg = aggregate(per, ["A", "B", "C", "D"])
    assert agg["by_type"]["A"]["f1"] == 1.0
    assert agg["micro"]["tp"] == 2 and agg["micro"]["fp"] == 1 and agg["micro"]["fn"] == 1
    assert agg["macro_f1"] == pytest.approx(0.25)  # only A scores; B, C, D are 0


def test_prf_zero_division():
    assert PRF().f1 == 0.0

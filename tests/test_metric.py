import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.metric import Score, normalise, score  # noqa: E402


def e(label, value):
    return {"label": label, "value": value}


def test_perfect_match():
    s = score([[e("EMAIL", "a@b.io"), e("CITY", "Leeds")]], [[e("CITY", "Leeds"), e("EMAIL", "a@b.io")]])
    assert s.micro.f1 == 1.0, "order must not matter"


def test_duplicate_values_need_duplicate_predictions():
    """The multiset behaviour: two identical gold entities require two predictions."""
    gold = [[e("GIVENNAME", "Alex"), e("GIVENNAME", "Alex")]]
    s = score(gold, [[e("GIVENNAME", "Alex")]])
    assert s.micro.tp == 1 and s.micro.fn == 1
    assert s.micro.recall == 0.5, "set-based scoring would wrongly give 1.0 here"


def test_wrong_label_right_value():
    s = score([[e("CITY", "Reading")]], [[e("SURNAME", "Reading")]])
    assert s.micro.tp == 0 and s.micro.fp == 1 and s.micro.fn == 1
    assert s.per_label["CITY"].recall == 0.0
    assert s.per_label["SURNAME"].precision == 0.0


def test_empty_prediction_is_all_misses():
    s = score([[e("EMAIL", "a@b.io"), e("AGE", "41")]], [[]])
    assert s.micro.fn == 2 and s.micro.tp == 0
    assert s.micro.precision == 0.0 and s.micro.recall == 0.0


def test_empty_gold_and_pred_scores_nothing():
    s = score([[]], [[]])
    assert s.micro.tp == s.micro.fp == s.micro.fn == 0
    assert s.n_examples == 1


def test_normalisation():
    assert normalise("  Sarah   Chen ") == "sarah chen"
    s = score([[e("GIVENNAME", "Sarah  Chen")]], [[e("GIVENNAME", "sarah chen")]])
    assert s.micro.f1 == 1.0


def test_schema_invalid_counts_and_zeroes_the_example():
    s = Score()
    s.add([e("EMAIL", "a@b.io")], None)
    assert s.schema_invalid == 1
    assert s.micro.fn == 1 and s.micro.tp == 0


def test_hallucinated_value_flagged():
    s = Score()
    s.add([e("EMAIL", "a@b.io")], [e("EMAIL", "ghost@nowhere.io")], source_text="Mail a@b.io today")
    assert s.hallucinated == 1, "predicted value absent from source is an escalation trigger"


def test_hallucination_check_ignores_case_and_spacing():
    s = Score()
    s.add([e("GIVENNAME", "Sarah")], [e("GIVENNAME", "sarah")], source_text="Call Sarah now")
    assert s.hallucinated == 0

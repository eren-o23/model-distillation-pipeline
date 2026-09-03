"""The Phase 4 verdict rests on a bootstrap interval, so the interval gets a test.

Runs locally in milliseconds — no GPU, no API, no fixtures. What it pins is the property the report
depends on: an interval that excludes zero only when the difference is real.
"""

from src.pii.metric import bootstrap_delta, micro_f1, row_counts

SEED = 20260825


def gold(n: int) -> list[list[dict]]:
    return [[{"label": "EMAIL", "value": f"a{i}@x.com"}, {"label": "CITY", "value": f"town{i}"}]
            for i in range(n)]


def test_micro_f1_matches_hand_computation():
    # 3 of 4 predicted entities right, 1 of 4 gold entities missed: P = R = F1 = 0.75.
    counts = [(3, 1, 1)]
    assert micro_f1(counts) == 0.75


def test_micro_f1_resamples_by_index():
    counts = [(1, 0, 0), (0, 0, 1)]  # one perfect row, one total miss
    assert micro_f1(counts, [0, 0, 0]) == 1.0
    assert micro_f1(counts, [1, 1]) == 0.0


def test_row_counts_scores_a_perfect_and_an_empty_prediction():
    g = gold(2)
    assert row_counts(g, [g[0], []]) == [(2, 0, 0), (0, 0, 2)]


def test_identical_models_give_an_interval_containing_zero():
    g = gold(200)
    preds = [row[:1] for row in g]  # both models make exactly the same predictions
    counts = row_counts(g, preds)
    lo, hi, ahead = bootstrap_delta(counts, counts, 400, SEED)
    assert lo == hi == 0.0
    assert ahead == 0.0


def test_a_clearly_better_model_gives_an_interval_above_zero():
    g = gold(200)
    good = row_counts(g, g)                       # perfect
    bad = row_counts(g, [row[:1] for row in g])   # finds half the entities
    lo, hi, ahead = bootstrap_delta(good, bad, 400, SEED)
    assert lo > 0, "an interval that spans zero here would call a real gap indistinguishable"
    assert ahead == 1.0


def test_a_null_result_is_reported_as_one():
    """One differing row in 200 must not clear the noise floor; twenty must.

    This is the boundary the whole Phase 4 verdict turns on — D-027 was called null on a 0.005 gap, and
    the report says "matches" or "beats" purely on whether this interval contains zero.
    """
    g = gold(200)
    perfect = row_counts(g, g)

    one_off = row_counts(g, [[] if i == 0 else row for i, row in enumerate(g)])
    lo, hi, _ = bootstrap_delta(one_off, perfect, 400, SEED)
    assert lo <= 0 <= hi, f"one differing row should not be a result: [{lo}, {hi}]"

    twenty_off = row_counts(g, [[] if i < 20 else row for i, row in enumerate(g)])
    lo, hi, _ = bootstrap_delta(twenty_off, perfect, 400, SEED)
    assert hi < 0, f"a 10% difference should clear the noise floor: [{lo}, {hi}]"

"""The break-even volume is the deliverable, so the arithmetic behind it gets a test.

Runs locally in milliseconds — no GPU, no API, no server. What it pins is the property the report
turns on: that the new formula agrees with Phase 4's published conclusion on Phase 4's own numbers
before anyone trusts it on Phase 5's.
"""

from src.pii.economics import (blended_per_1k, break_even_per_day, capacity_fraction,
                               capacity_per_day, per_1k)
from src.pii.router import Completion, blend, quantile, should_escalate

# Phase 4, as published: 0.336 req/s on a $0.526/hr T4 against a teacher measured at $0.4366/1,000.
P4_RPS, P4_RATE = 0.336, 0.526
TEACHER_PER_REQUEST = 0.4366 / 1000


def test_per_1k_reproduces_the_phase4_cost_table():
    """The number in reports/phase4.md's cost row, to the precision it is printed at."""
    assert round(per_1k(P4_RATE, P4_RPS), 2) == 0.43
    # The utilisation columns charge for idle time rather than discounting it: half the traffic on the
    # same hourly card is twice the cost per request, not half.
    assert round(per_1k(P4_RATE, P4_RPS, 0.5), 2) == 0.87
    assert round(per_1k(P4_RATE, P4_RPS, 0.25), 2) == 1.74


def test_idle_and_dead_cards_cost_infinity_rather_than_dividing_by_zero():
    assert per_1k(P4_RATE, 0.0) == float("inf")
    assert capacity_fraction(P4_RATE, TEACHER_PER_REQUEST, 0.0) == float("inf")


def test_phase4_had_no_usable_break_even():
    """D-032's verdict, recovered from the formula rather than restated.

    The strict inequality alone would have called Phase 4 a win — the student was $0.4349 against the
    API's $0.4366. What makes it not a win is *where* the break-even sits: at 99.6% of what the card
    can serve in a day, so it only pays on a GPU that is never idle for a second. That margin is the
    result, which is why the function returns it instead of a bool.
    """
    fraction = capacity_fraction(P4_RATE, TEACHER_PER_REQUEST, P4_RPS)
    assert 0.99 < fraction < 1.0, f"Phase 4 should sit right at the capacity ceiling, got {fraction}"


def test_hitting_the_phase5_throughput_target_creates_real_headroom():
    """D-032 named 6.7 req/s as the tenfold-advantage target. It must land far below the ceiling."""
    fraction = capacity_fraction(P4_RATE, TEACHER_PER_REQUEST, 6.7)
    assert fraction < 0.1, f"20x the throughput should leave an order of magnitude spare, got {fraction}"


def test_escalation_raises_the_break_even_volume():
    """The router's premium must be visible, not absorbed.

    Every escalated request is one the GPU was paid for *and* the API is billed for, so the saving per
    request shrinks by exactly the escalation rate and the volume needed to amortise the card rises.
    A formula that ignored this would quote a break-even the router cannot actually reach.
    """
    alone = break_even_per_day(P4_RATE, TEACHER_PER_REQUEST)
    routed = break_even_per_day(P4_RATE, TEACHER_PER_REQUEST, escalation_rate=0.15)
    assert routed > alone
    # 15% escalation leaves 85% of the saving, so the break-even volume scales by 1/0.85.
    assert round(routed / alone, 4) == round(1 / 0.85, 4)


def test_escalating_everything_never_breaks_even():
    """At 100% escalation the GPU is pure overhead — it is paid for and the API is billed anyway."""
    assert break_even_per_day(P4_RATE, TEACHER_PER_REQUEST, escalation_rate=1.0) == float("inf")


def test_blended_cost_charges_the_gpu_on_every_request():
    """An escalated request still ran on the student first. Escalation is not a discount on the card."""
    gpu_only = per_1k(P4_RATE, P4_RPS)
    blended = blended_per_1k(P4_RATE, P4_RPS, TEACHER_PER_REQUEST, escalation_rate=0.2)
    assert round(blended - gpu_only, 6) == round(1000 * 0.2 * TEACHER_PER_REQUEST, 6)


def test_capacity_is_a_day_of_seconds():
    assert capacity_per_day(1.0) == 86_400


def test_quantile_endpoints_are_the_two_pure_arms():
    """q=0 must escalate nothing and q=1 everything, exactly — the sweep is bounded by these."""
    conf = [-3.0, -2.0, -1.0, -0.5]
    assert all(c >= quantile(conf, 0.0) for c in conf)      # nothing is below -inf
    assert all(c < quantile(conf, 1.0) for c in conf)       # everything is below +inf
    # A quarter of four rows is the single worst one.
    assert sum(c < quantile(conf, 0.25) for c in conf) == 1


def test_schema_invalid_escalates_whatever_the_threshold_says():
    """The hard trigger. An unparseable answer detects nothing, so confidence in it is irrelevant."""
    confident_garbage = Completion("not json", None, -0.001, -0.001, 3)
    assert should_escalate(confident_garbage, float("-inf"))


def test_a_confident_valid_answer_is_never_escalated_by_the_student_only_arm():
    good = Completion("{}", [{"label": "EMAIL", "value": "a@b.c"}], -0.01, -2.5, 10)
    assert not should_escalate(good, float("-inf"))
    assert should_escalate(good, float("inf"))
    # The signal is selectable, and the two disagree on this row by construction: min is -2.5, mean -0.01.
    assert should_escalate(good, -1.0, signal="min")
    assert not should_escalate(good, -1.0, signal="mean")


def test_an_empty_completion_escalates_rather_than_reading_as_certain():
    """No tokens means no evidence. Defaulting to 0.0 here would route silence straight to the user."""
    assert should_escalate(Completion("", [], float("-inf"), float("-inf"), 0), -99.0)


def test_blend_takes_the_teacher_only_where_it_was_called():
    assert blend(["s0", "s1", "s2"], ["t0", "t1", "t2"], [False, True, False]) == ["s0", "t1", "s2"]

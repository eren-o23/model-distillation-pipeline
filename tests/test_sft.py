"""Checks for the Phase 2 training-set build.

Covers the three pieces of logic that can silently corrupt the training file: template dedup, the
hallucination strip, and the exact bytes of the assistant turn.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import DATA_DIR, template_key  # noqa: E402
from src.pii.metric import strip_hallucinated  # noqa: E402
from src.pii.student import SHORT_SYSTEM, to_sft_example  # noqa: E402
from src.pii.teacher import _parse  # noqa: E402

RAW_LOG = Path(__file__).resolve().parents[1] / "reports" / "raw" / "train-teacher.jsonl"


def e(label, value):
    return {"label": label, "value": value}


def test_template_key_collapses_same_carrier_sentence():
    a = {"source_text": "Contact Alex Smith at alex@b.io", "entities": [e("GIVENNAME", "Alex"), e("SURNAME", "Smith"), e("EMAIL", "alex@b.io")]}
    b = {"source_text": "Contact Priya Rao at priya@c.io", "entities": [e("GIVENNAME", "Priya"), e("SURNAME", "Rao"), e("EMAIL", "priya@c.io")]}
    c = {"source_text": "Invoice for Alex Smith", "entities": [e("GIVENNAME", "Alex"), e("SURNAME", "Smith")]}
    assert template_key(a) == template_key(b), "same template, different values, must collapse"
    assert template_key(a) != template_key(c), "different carrier text must not collapse"


def test_strip_hallucinated_keeps_siblings():
    text = "Alex Smith lives in Leeds"
    kept = strip_hallucinated([e("GIVENNAME", "Alex"), e("CITY", "Paris"), e("SURNAME", "Smith")], text)
    assert kept == [e("GIVENNAME", "Alex"), e("SURNAME", "Smith")], "drop only the invented entity"


def test_sft_example_round_trips_through_the_teacher_parser():
    """The assistant turn must be exactly what the student is expected to emit at serving time, so it
    has to survive the same validator Phase 4/5 will score its output with."""
    ents = [e("GIVENNAME", "Ana"), e("CITY", "Košice")]
    row = to_sft_example("Ana lives in Košice", ents)
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][0]["content"] == SHORT_SYSTEM
    assert _parse(row["messages"][2]["content"]) == ents
    # Compact separators and no \u escaping, asserted as literal bytes: this is what the student learns
    # to emit, so a formatting change here is a silent retrain, not a cosmetic diff.
    assert row["messages"][2]["content"] == (
        '{"entities":[{"label":"GIVENNAME","value":"Ana"},{"label":"CITY","value":"Košice"}]}'
    )


def test_no_test_split_uid_leaks_into_the_teacher_run():
    """Cheap insurance on the sealed test set.

    Needs both halves of the comparison, and they have different lifetimes: the teacher log is committed,
    while `data/*.jsonl` is gitignored. On Kaggle the log therefore exists and the sealed split does not,
    so checking only the log — as this did — turned a missing precondition into a failure.

    Skipping rather than returning, so a run where this check did not happen says so instead of reporting
    a pass it never earned.
    """
    test_split = DATA_DIR / "test.jsonl"
    if not RAW_LOG.exists() or not test_split.exists():
        pytest.skip("needs the teacher log and the sealed test split; the split is absent by design on Kaggle")
    test_uids = {json.loads(line)["uid"] for line in test_split.open()}
    used = {json.loads(line)["uid"] for line in RAW_LOG.open()}
    assert not (used & test_uids), "test split must never be sent to the teacher"

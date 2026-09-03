"""Phase 5: what escalation buys, swept offline over the confidence threshold.

**This script cannot spend money.** It calls nothing. Every arm of the sweep is a lookup against two
measurements already on disk: the student's per-row answers and logprobs from `benchmark_vllm.py
--parity`, and the teacher's per-row answers from `benchmark_teacher.py --split val --yes`. Escalating
a row means substituting the teacher's cached prediction for the student's, which makes the whole
threshold sweep arithmetic and re-runnable at no cost.

  python scripts/benchmark_router.py

The sweep is parameterised by **target escalation rate**, not by raw logprob, because that is the knob
an operator actually has: "I will pay the API for x% of traffic, what does it buy?" The threshold is
recovered as the matching quantile of the observed confidence distribution.

Every operating point is measured against **random escalation at the same rate**. Without that column
the sweep cannot distinguish a confidence signal that works from the fact that escalating anything to a
0.822-F1 teacher helps a bit. If the signal does not beat random, the honest report says the router is
a cost with no discrimination behind it.

Developed on `val`, necessarily: the test set was opened once in Phase 4 and is spent (D-028).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import SEED, load_split, verify_frozen  # noqa: E402
from src.pii.economics import blended_per_1k  # noqa: E402
from src.pii.metric import Score  # noqa: E402
from src.pii.router import blend, quantile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports" / "raw" / "phase5"

# The operator-facing knob. 0.0 is the student alone (only schema failures escalate) and 1.0 is the
# teacher alone — both endpoints are in the sweep so the curve is bounded by the two Phase 4 numbers
# it has to sit between.
TARGET_RATES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
SIGNALS = ("min_logprob", "mean_logprob")


def measure(gold_rows: list[dict], preds: list) -> Score:
    s = Score()
    for row, pred in zip(gold_rows, preds, strict=True):
        s.add(row["entities"], pred, row["source_text"])
    return s


def operating_point(gold_rows, student_rows, teacher_preds, escalate, teacher_per_request,
                    rps, rate) -> dict:
    s = measure(gold_rows, blend([r["entities"] for r in student_rows], teacher_preds, escalate))
    n = len(escalate)
    e = sum(escalate) / n
    idcard = s.per_label.get("IDCARDNUM")
    point = {
        "escalation_rate": e,
        "escalated": sum(escalate),
        "micro_f1": s.micro.f1,
        "micro_precision": s.micro.precision,
        "micro_recall": s.micro.recall,
        "schema_invalid": s.schema_invalid,
        "idcardnum_recall": idcard.recall if idcard else None,
        "idcardnum_f1": idcard.f1 if idcard else None,
    }
    if rps and rate:
        point["usd_per_1k"] = blended_per_1k(rate, rps, teacher_per_request, e)
    return point


def main() -> None:
    student_path = next(iter(sorted(RAW.glob("vllm-parity-val-*.json"))), None)
    teacher_path = next(iter(sorted(RAW.glob("teacher-val-*.json"))), None)
    if not student_path:
        sys.exit("no student parity run — scripts/benchmark_vllm.py --parity has not been run")
    if not teacher_path:
        sys.exit("no teacher val run — scripts/benchmark_teacher.py --split val --yes has not been run")

    student = json.loads(student_path.read_text())
    teacher = json.loads(teacher_path.read_text())

    split_sha = verify_frozen("val")
    rows = load_split("val")
    # Both arms must have been measured against the file just verified, and in the same row order.
    # Without this a rebuilt split would silently blend two different datasets.
    for name, blob in (("student", student), ("teacher", teacher)):
        assert blob["split_sha256"] == split_sha, f"{name} was measured against a different val split"
    n = min(student["n"], teacher["n"])
    rows = rows[:n]
    assert student["uids"][:n] == [r["uid"] for r in rows], "student rows are out of order"
    assert teacher["uids"][:n] == [r["uid"] for r in rows], "teacher rows are out of order"

    student_rows = student["rows"][:n]
    teacher_preds = teacher["predictions"][:n]
    teacher_per_request = teacher["cost_usd"] / teacher["n"]

    # Throughput at the concurrency that maximises it, matching how Phase 4 picked its cost basis.
    lat_path = next(iter(sorted(RAW.glob("vllm-latency-*.json"))), None)
    rps = rate = None
    if lat_path:
        lat = json.loads(lat_path.read_text())
        by_c = lat["by_concurrency"]
        best = max(by_c, key=lambda c: by_c[c]["requests_per_s"])
        rps, rate = by_c[best]["requests_per_s"], lat["gpu_rate_usd_h"]
        print(f"cost basis: {rps:.2f} req/s at concurrency {best}, ${rate}/hr")

    schema_bad = [r["entities"] is None for r in student_rows]
    print(f"\n{n} val rows · {sum(schema_bad)} schema-invalid (escalate unconditionally) · "
          f"teacher ${teacher_per_request * 1000:.2f}/1k\n")

    rng = random.Random(SEED)
    out = {"tag": "router-sweep", "split": "val", "split_sha256": split_sha, "n": n,
           "teacher_per_request": teacher_per_request, "requests_per_s": rps, "gpu_rate_usd_h": rate,
           "student_only_f1": measure(rows, [r["entities"] for r in student_rows]).micro.f1,
           "teacher_only_f1": measure(rows, teacher_preds).micro.f1,
           "by_signal": {}}

    for signal in SIGNALS:
        conf = [r[signal] for r in student_rows]
        points, randoms = [], []
        print(f"--- {signal} " + "-" * 52)
        print(f"{'target':>7} {'actual':>7} {'thresh':>9} {'micro-F1':>9} {'IDCARD R':>9} "
              f"{'random F1':>10}")
        for q in TARGET_RATES:
            thr = quantile(conf, q)
            escalate = [bad or c < thr for bad, c in zip(schema_bad, conf, strict=True)]
            pt = operating_point(rows, student_rows, teacher_preds, escalate, teacher_per_request,
                                 rps, rate) | {"target_rate": q, "threshold": thr}

            # Same number of escalations, chosen at random. The control that says whether the
            # confidence signal is doing any work at all.
            k = sum(escalate)
            idx = set(rng.sample(range(n), k))
            rnd = operating_point(rows, student_rows, teacher_preds,
                                  [i in idx for i in range(n)], teacher_per_request, rps, rate)
            pt["random_micro_f1"] = rnd["micro_f1"]
            pt["random_idcardnum_recall"] = rnd["idcardnum_recall"]

            points.append(pt)
            randoms.append(rnd)
            print(f"{q:>7.0%} {pt['escalation_rate']:>7.1%} "
                  f"{'-inf' if thr == float('-inf') else ('inf' if thr == float('inf') else f'{thr:.3f}'):>9} "
                  f"{pt['micro_f1']:>9.3f} {pt['idcardnum_recall'] or 0:>9.3f} "
                  f"{pt['random_micro_f1']:>10.3f}")
        out["by_signal"][signal] = points
        print()

    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "router-sweep.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote reports/raw/phase5/{path.name}")


if __name__ == "__main__":
    main()

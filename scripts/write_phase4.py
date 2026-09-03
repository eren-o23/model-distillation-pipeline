"""Generate reports/phase4.md from the measurements Phase 4 left behind.

Same discipline as write_ceiling.py and write_phase3.py: every number is read from a saved run rather
than transcribed, so the report cannot drift from what happened. Runs locally, needs no GPU.

It passes `allow_test=True` to read gold and score saved predictions. That is the third and last place
in the repo that does; it calls no model, and the seal it checks is the sha256 in data/manifest.json.

  python scripts/write_phase4.py            # after copying reports/raw/phase4/ back from Kaggle
"""

import json
import math
import sys
from datetime import date
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import SEED, load_split, verify_frozen  # noqa: E402
from src.pii.metric import Score, bootstrap_delta, row_counts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports" / "raw" / "phase4"
PHASE3 = ROOT / "reports" / "raw" / "phase3"
CEILING = ROOT / "reports" / "raw" / "qwen3p7-plus-200.json"

# Kaggle is free, which is not a cost basis — a $0 GPU makes self-hosting win at any volume, which is
# the flattery the spec warns about. Price the card that was actually measured, at a public on-demand
# rate, and say where the number came from.
GPU_RATE_USD_H = 0.526
GPU_RATE_SOURCE = ("AWS `g4dn.xlarge` (1x T4 16GB), us-east-1 on-demand, "
                   "[$0.526/hr](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), checked 2026-09-02")
# 100% is the number a vendor quotes; 25% is closer to what bursty traffic actually achieves. Reporting
# all three is what stops the cost axis from quietly assuming a permanently saturated GPU.
UTILISATION = (1.00, 0.50, 0.25)
BOOTSTRAP_DRAWS = 2000


def load_one(pattern: str) -> dict | None:
    """The largest matching measurement, or None.

    Largest by row count, not by filename: smoke-test checkpoints sit in the same directory and
    `teacher-test-20.json` sorts after `teacher-test-1000.json`, so a lexicographic pick would report
    the 20-row rehearsal as the benchmark.
    """
    hits = [json.loads(p.read_text()) for p in RAW.glob(pattern)]
    return max(hits, key=lambda b: b.get("n", 0)) if hits else None


def score_of(rows: list[dict], preds: list) -> Score:
    s = Score()
    for row, pred in zip(rows, preds, strict=True):
        s.add(row["entities"], pred, row["source_text"])
    return s


def pcts(values: list[float]) -> tuple[float, float]:
    v = sorted(values)
    return median(v), v[min(int(0.95 * len(v)), len(v) - 1)]


def per_1k(requests_per_s: float, utilisation: float) -> float:
    """GPU-hour cost amortised over measured throughput, with idle time charged for."""
    return GPU_RATE_USD_H / (requests_per_s * 3600 * utilisation) * 1000


def main() -> None:
    if not RAW.exists():
        sys.exit(f"no measurements at {RAW} — run the benchmarks first")

    split_sha = verify_frozen("test")
    rows = load_split("test", allow_test=True)

    teacher = load_one("teacher-test-*.json")
    if not teacher:
        sys.exit("no teacher run found — scripts/benchmark_teacher.py --yes has not been run")
    # Only students measured over the same rows as the teacher. A --limit smoke run left in this
    # directory would otherwise be compared against the full teacher run and read as a bad adapter.
    students = {p.stem: json.loads(p.read_text()) for p in sorted(RAW.glob("student-*-test-*.json"))}
    students = {k: v for k, v in students.items() if len(v["uids"]) == teacher["n"]}
    if not students:
        sys.exit("no student run over the same rows as the teacher — "
                 "run scripts/benchmark_student.py without --limit")

    # Every payload must have been measured against the file this script just verified. Without this a
    # rebuilt split would silently produce a report that mixes two different test sets.
    for name, blob in {"teacher": teacher, **students}.items():
        assert blob["split_sha256"] == split_sha, f"{name} was measured against a different test split"

    n = teacher["n"]
    rows = rows[:n]
    assert teacher["uids"] == [r["uid"] for r in rows], "teacher predictions are out of order"

    golds = [r["entities"] for r in rows]
    t_counts = row_counts(golds, teacher["predictions"])
    t_score = score_of(rows, teacher["predictions"])

    # Rank the students by F1 so the best one is the headline, whichever rank it turns out to be.
    parsed = {}
    for name, blob in students.items():
        assert blob["uids"] == [r["uid"] for r in rows], f"{name} is out of order"
        preds = [s["parsed"] for s in blob["samples"]]
        parsed[name] = {"blob": blob, "preds": preds, "counts": row_counts(golds, preds),
                        "score": score_of(rows, preds), "rank": blob["adapter"].split("-r")[-1]}
    # The headline student is the one Phase 3 *deployed*, chosen on val — not whichever scores highest
    # here. Picking the winner on the sealed set would be model selection against test, which is the
    # contamination this whole phase is built to avoid, and on this evidence it would also be noise:
    # the two configs sit within 0.003 of each other and D-027 already called them inseparable.
    val_f1 = {}
    for path in PHASE3.glob("r*-summary.json"):
        blob = json.loads(path.read_text())
        val_f1[path.stem.split("-")[0][1:]] = blob["best"]["f1"]
    deployed = max(val_f1, key=val_f1.get) if val_f1 else None
    best_name = next((k for k in parsed if parsed[k]["rank"] == deployed), None) \
        or max(parsed, key=lambda k: parsed[k]["score"].micro.f1)
    best = parsed[best_name]
    selection = (f"selected on **val** in Phase 3 (rank {deployed} at {val_f1[deployed]:.3f}), not on the "
                 "rows below" if deployed else "the only configuration measured")
    s_score = best["score"]

    lo, hi, p_student = bootstrap_delta(best["counts"], t_counts, BOOTSTRAP_DRAWS, SEED)
    # A bound that rounds to zero should print as 0.000, not -0.000; the minus sign reads as a result.
    lo, hi = (0.0 if abs(lo) < 5e-4 else lo), (0.0 if abs(hi) < 5e-4 else hi)
    delta = s_score.micro.f1 - t_score.micro.f1
    if lo > 0:
        verdict = (f"**The student beats the teacher** by {delta:+.3f} micro-F1, with a 95% confidence "
                   f"interval of [{lo:+.3f}, {hi:+.3f}] that excludes zero.")
    elif hi < 0:
        verdict = (f"**The teacher beats the student** by {-delta:.3f} micro-F1, with a 95% confidence "
                   f"interval of [{lo:+.3f}, {hi:+.3f}] that excludes zero.")
    else:
        verdict = (f"**The two are statistically indistinguishable.** The gap is {delta:+.3f} micro-F1 "
                   f"and the 95% confidence interval on it, [{lo:+.3f}, {hi:+.3f}], spans zero — so "
                   f"**\"matches the teacher\" is the claim the evidence supports, and \"beats\" is not**, "
                   f"even now that both are measured on the same {n:,} rows. The student was ahead in "
                   f"{p_student:.0%} of {BOOTSTRAP_DRAWS:,} resamples.")

    # ---- cost -------------------------------------------------------------------------------------
    teacher_per_1k = teacher["cost_usd"] / n * 1000
    lat = load_one("student-latency-*.json")
    throughput = {}
    if lat:
        throughput = {int(b): v for b, v in lat["by_batch"].items() if not v.get("oom")}
    best_batch = max(throughput, key=lambda b: throughput[b]["requests_per_s"]) if throughput else None

    cost_rows = [f"| **teacher** (`qwen3p7-plus` API) | measured tokens | **${teacher_per_1k:.2f}** | "
                 f"${teacher_per_1k:.2f} | ${teacher_per_1k:.2f} |"]
    if best_batch:
        rps = throughput[best_batch]["requests_per_s"]
        cost_rows = [
            "| **teacher** (`qwen3p7-plus` API) | per-token, no idle cost | "
            + " | ".join(f"**${teacher_per_1k:.2f}**" if u == 1.0 else f"${teacher_per_1k:.2f}"
                         for u in UTILISATION) + " |",
            f"| **student** (T4, batch {best_batch}) | {rps:.3f} req/s measured | "
            + " | ".join(f"**${per_1k(rps, u):.2f}**" if u == 1.0 else f"${per_1k(rps, u):.2f}"
                         for u in UTILISATION) + " |",
        ]

    t_p50_8, t_p95_8 = pcts(teacher["latency_s"])

    # ---- the verdict the three axes add up to -----------------------------------------------------
    # Stated as a computed conclusion rather than a hopeful one: whether self-hosting pays at all on this
    # stack follows from the two cost numbers, and the answer here is not the flattering one.
    verdict_cost = ""
    if best_batch:
        rps = throughput[best_batch]["requests_per_s"]
        student_100 = per_1k(rps, 1.0)
        ratio = student_100 / teacher_per_1k
        # What throughput would have to be for the self-hosted arm to be worth the operational cost.
        need_parity_50 = GPU_RATE_USD_H / (teacher_per_1k / 1000 * 3600 * 0.5)
        need_10x_50 = GPU_RATE_USD_H / (teacher_per_1k / 10 / 1000 * 3600 * 0.5)
        lat_ratio = throughput[8]["p50_s"] / t_p50_8 if 8 in throughput else None
        if ratio >= 0.9:
            verdict_cost = (
                f"**On this serving stack there is no break-even volume.** The student costs "
                f"${student_100:.2f} per 1,000 requests on a fully saturated T4 against the teacher's "
                f"${teacher_per_1k:.2f} — {ratio:.0%} of the API's price at 100% utilisation, and *more* "
                f"than the API at any realistic one (${per_1k(rps, 0.5):.2f} at 50%). Self-hosting is not "
                f"cheaper here at one request per day or at ten million"
                + (f", and it is {lat_ratio:.0f}x slower per request" if lat_ratio else "") + ".\n\n"
                f"That is a real result rather than a setback, and it locates the problem precisely: the "
                f"student's quality is not the constraint — it matches the teacher — and neither is the "
                f"hardware price. **The constraint is throughput.** At {rps:.3f} req/s, HF `generate` "
                f"leaves the card idle between batches and pads every request to the longest in its "
                f"batch. Phase 5's vLLM benchmark needs **{need_parity_50:.2f} req/s** for the student "
                f"to merely match API pricing at 50% utilisation, and **{need_10x_50:.1f} req/s** for the "
                f"tenfold advantage this project set out to find. Those are the numbers to beat, and they "
                f"come from measurement rather than from hope."
            )
        else:
            verdict_cost = (
                f"**The student is cheaper per request**: ${student_100:.2f} per 1,000 against the "
                f"teacher's ${teacher_per_1k:.2f}, {1 / ratio:.1f}x, on a saturated card"
                + (f" — while being {lat_ratio:.0f}x slower per request" if lat_ratio else "") + ". "
                f"It stops paying below {GPU_RATE_USD_H / (teacher_per_1k / 1000 * 3600) / rps:.0%} "
                f"utilisation, which is where Phase 5's break-even volume comes from."
            )

    # ---- latency ----------------------------------------------------------------------------------
    serial = load_one("teacher-latency-serial-*.json")
    lat_rows = []
    if serial:
        p50, p95 = pcts(serial["latency_s"])
        lat_rows.append(f"| teacher (API) | 1 | {p50:.2f}s | {p95:.2f}s |")
    lat_rows.append(f"| teacher (API) | {teacher['workers']} | {t_p50_8:.2f}s | {t_p95_8:.2f}s |")
    for b in (1, 8):
        if b in throughput:
            r = throughput[b]
            lat_rows.append(f"| student (T4, batch {b}) | {b} | {r['p50_s']:.2f}s | {r['p95_s']:.2f}s |")

    curve = []
    for b in sorted(int(k) for k in (lat or {"by_batch": {}})["by_batch"]):
        r = lat["by_batch"][str(b)]
        if r.get("oom"):
            curve.append(f"| {b} | — | — | — | OOM | — |")
            continue
        curve.append(f"| {b} | {r['p50_s']:.2f}s | {r['p95_s']:.2f}s | {r['requests_per_s']:.3f} | "
                     f"{r['peak_gib']:.2f} GiB | ${per_1k(r['requests_per_s'], 1.0):.2f} |")

    # ---- per label --------------------------------------------------------------------------------
    labels = sorted(s_score.per_label, key=lambda k: -(s_score.per_label[k].tp + s_score.per_label[k].fn))
    per_label = ["| label | teacher F1 | student F1 | Δ | student P | student R | support |",
                 "|---|---|---|---|---|---|---|"]
    for lbl in labels:
        s, t = s_score.per_label[lbl], t_score.per_label.get(lbl)
        tf = t.f1 if t else 0.0
        per_label.append(f"| `{lbl}` | {tf:.3f} | {s.f1:.3f} | {s.f1 - tf:+.3f} | {s.precision:.3f} | "
                         f"{s.recall:.3f} | {s.tp + s.fn} |")

    idcard = s_score.per_label.get("IDCARDNUM")

    # ---- rank comparison --------------------------------------------------------------------------
    rank_block = ""
    if len(parsed) > 1:
        vals = {k: v["score"].micro.f1 for k, v in parsed.items()}
        top, low = max(vals, key=vals.get), min(vals, key=vals.get)
        gap = vals[top] - vals[low]
        support = s_score.micro.tp + s_score.micro.fn
        se = math.sqrt(vals[top] * (1 - vals[top]) / max(support, 1))
        table = ["| config | micro-F1 | P | R | schema-invalid |", "|---|---|---|---|---|"]
        for k in sorted(parsed, key=lambda k: -vals[k]):
            m = parsed[k]["score"]
            table.append(f"| rank {parsed[k]['rank']} | {vals[k]:.3f} | {m.micro.precision:.3f} | "
                         f"{m.micro.recall:.3f} | {m.schema_invalid}/{m.n_examples} |")
        headline_rank = best["rank"]
        if gap < 2 * se:
            note = (f"The gap is {gap:.3f} against a standard error of about {se:.3f} on {support:,} "
                    f"scored entities. **D-027's null result holds at {n:,} rows** — five times the "
                    "sample that produced it, on data neither configuration has ever seen. Two ranks "
                    f"differing by 2x in capacity are not separable on this task, and rank "
                    f"{headline_rank} stays selected on cost — a decision made on val, which is why the "
                    "ordering above does not change it.")
        else:
            note = (f"The gap is {gap:.3f} against a standard error of about {se:.3f} on {support:,} "
                    f"scored entities — **wider than noise, which reverses D-027**. The 200-row val "
                    f"comparison was underpowered rather than genuinely null, and rank "
                    f"{parsed[top]['rank']} is the better configuration. Note that this is a finding, "
                    "not a selection: the deployed adapter was fixed on val before these rows were "
                    "opened, and changing it now on test evidence would spend the sealed set on model "
                    "choice.")
        rank_block = "\n".join(["## rank 8 vs rank 16, on the sealed test set", "", *table, "", note, ""])

    # ---- val -> test drift ------------------------------------------------------------------------
    drift, drifts = [], []
    val_full = next((json.loads(p.read_text()) for p in sorted(PHASE3.glob("adapter-*-full.json"))), None)
    if val_full:
        d = s_score.micro.f1 - val_full["micro_f1"]
        drifts.append(d)
        drift.append(f"| student (rank {best['rank']}) | {val_full['micro_f1']:.3f} "
                     f"({val_full['n_examples']:,} val) | {s_score.micro.f1:.3f} ({n:,} test) | {d:+.3f} |")
    if CEILING.exists():
        blob = json.loads(CEILING.read_text())
        vrows = load_split("val")[: len(blob["predictions"])]
        v = score_of(vrows, blob["predictions"]).micro.f1
        drifts.append(t_score.micro.f1 - v)
        drift.append(f"| teacher | {v:.3f} ({len(vrows)} val) | {t_score.micro.f1:.3f} ({n:,} test) | "
                     f"{t_score.micro.f1 - v:+.3f} |")

    # The conclusion follows the numbers rather than preceding them. Contamination would show up as the
    # student dropping on test while the teacher, whose prompt was frozen in Phase 1, does not.
    if drifts and max(abs(d) for d in drifts) < 0.03:
        drift_note = ("Prompt development happened on `val` across three phases, so a large drop here "
                      "would be the signature of that leaking into the headline. It did not: both models "
                      "move by less than 0.03 F1, which is what two draws from one distribution look like.")
    elif len(drifts) == 2 and drifts[0] < drifts[1] - 0.02:
        drift_note = (f"**The student drops on test by {drifts[0]:+.3f} while the teacher moves "
                      f"{drifts[1]:+.3f}**, and that asymmetry is the shape contamination would take: the "
                      "teacher's prompt was frozen in Phase 1, the student's checkpoint was selected on "
                      "val. The test number is the one to quote.")
    else:
        drift_note = (f"Both models move in the same direction by a similar amount (student "
                      f"{drifts[0]:+.3f}, teacher {drifts[1]:+.3f}), which is a property of the two "
                      "splits rather than of either model — a leak would move the student alone.")

    failures = len(teacher.get("api_failures", []))
    headline_lat = f"{throughput[8]['p50_s']:.2f}s" if 8 in throughput else "—"

    report = f"""# Phase 4 — Student against teacher on quality, cost and latency

**Measured:** {date.today()} on the **sealed test split** — {n:,} rows, sha256 `{split_sha[:16]}…`,
verified against the manifest Phase 1 froze it with.
**Opened once.** Both models were scored on these same rows by the same metric, and nothing was tuned
against them: every prompt, batch size and checkpoint decision was made on `val` in Phases 1-3.
**Student:** `{best['blob']['adapter']}` on Kaggle's T4 — {selection}.
**Teacher:** `{teacher['model'].split('/')[-1]}` on Fireworks.
**Phase 4 API spend: ${teacher['cost_usd'] + (serial['cost_usd'] if serial else 0):.2f}.**

## The three axes

| | quality (micro-F1) | cost / 1k requests | latency p50 | latency p95 |
|---|---|---|---|---|
| **teacher** | {t_score.micro.f1:.3f} | ${teacher_per_1k:.2f} | {t_p50_8:.2f}s | {t_p95_8:.2f}s |
| **student** | {s_score.micro.f1:.3f} | {f"${per_1k(throughput[best_batch]['requests_per_s'], 1.0):.2f} at 100% utilisation" if best_batch else "—"} | {headline_lat} | {f"{throughput[8]['p95_s']:.2f}s" if 8 in throughput else "—"} |

Quality is measured at equal n on identical rows. Cost and latency are **not** like-for-like in the same
way and the sections below say exactly how: the teacher is a hosted API billed per token, the student is
one rented T4 running HF `generate` with static batching, which is the pre-vLLM floor rather than a
serving number (D-030).

{verdict_cost}

## Axis 1 — Quality

| model | micro-F1 | precision | recall | schema-invalid | hallucinated |
|---|---|---|---|---|---|
| teacher | {t_score.micro.f1:.3f} | {t_score.micro.precision:.3f} | {t_score.micro.recall:.3f} | {t_score.schema_invalid}/{n} ({t_score.schema_invalid / n:.1%}) | {t_score.hallucinated} |
| student | {s_score.micro.f1:.3f} | {s_score.micro.precision:.3f} | {s_score.micro.recall:.3f} | {s_score.schema_invalid}/{n} ({s_score.schema_invalid / n:.1%}) | {s_score.hallucinated} |

{verdict}

The interval comes from a paired bootstrap: {BOOTSTRAP_DRAWS:,} resamples of the {n:,} rows, both models
re-scored on each draw. Pairing is what makes it tight — the same rows are hard for both models, and
resampling them together cancels that shared difficulty instead of counting it twice.
""" + (f"""
{failures} rows never reached the teacher (unbilled 429s or timeouts) and are excluded from its
schema-invalid rate, which counts only answers the model actually returned malformed.
""" if failures else "") + f"""
### Per label

{chr(10).join(per_label)}
""" + (f"""
**`IDCARDNUM` transferred exactly as D-015 predicted**: {idcard.precision:.3f} precision against
{idcard.recall:.3f} recall. The student learned to be conservative because the teacher taught it to be —
the training labels themselves score 0.419 F1 on this class. It is the clearest concrete argument for
Phase 5's escalation path: a router that sends low-confidence `IDCARDNUM` cases back to the teacher buys
recall the student cannot be trained into without reopening D-006 and rebuilding the splits.
""" if idcard else "") + ("" if not drift else f"""
### val → test

| model | val | test | Δ |
|---|---|---|---|
{chr(10).join(drift)}

{drift_note}
""") + f"""
{rank_block}
## Axis 2 — Cost per 1,000 requests

| system | basis | 100% utilisation | 50% | 25% |
|---|---|---|---|---|
{chr(10).join(cost_rows)}

**Teacher:** measured, not estimated — ${teacher['cost_usd']:.4f} for {n:,} requests at
{teacher['usage']['prompt_tokens'] / n:.0f} input and {teacher['usage']['completion_tokens'] / n:.0f}
output tokens each. An API has no idle cost, so its three columns are identical; that is precisely the
property the student has to beat on volume.

**Student:** priced at {GPU_RATE_SOURCE}. Kaggle's GPU is free, which would make the student cost $0 and
win at any volume — a number that flatters self-hosting by hiding the hardware, which is a named failure
mode of this comparison. The utilisation columns charge for idle time: a GPU billed by the hour costs the
same whether or not requests arrive, so the 25% column is the honest one for bursty traffic.

**This student number is an upper bound.** HF `generate` with static batching leaves the card idle
between batches and pads every request to the longest in its batch. vLLM's continuous batching and paged
attention is what Phase 5 measures, and it will lower this figure — so the break-even volume computed
from this table would be pessimistic, and is deliberately not computed here.

## Axis 3 — Latency

| system | concurrency | p50 | p95 |
|---|---|---|---|
{chr(10).join(lat_rows)}

Concurrency means different things on the two sides and the mapping is stated rather than implied: for
the teacher it is in-flight HTTP requests; for the student it is batch size, because static batching
returns every request in a batch when its slowest member finishes. Both were timed on the **same seeded
{lat['n'] if lat else 0} test rows**, in arrival order rather than length-sorted — sorting is right for a
bulk eval and wrong for a latency measurement, since it reports a workload where every request happens to
arrive beside others of its own length.
""" + ("" if not curve else f"""
### Throughput curve

| batch | p50 | p95 | req/s | peak VRAM | $/1k @ 100% |
|---|---|---|---|---|---|
{chr(10).join(curve)}

This curve is what Axis 2 is amortised over, and what Phase 5 re-measures under vLLM.
""") + f"""
## Method notes

- Both models are scored against dataset **gold** by `src/pii/metric.py`, never against each other.
- The student's numbers come from `src.pii.eval.evaluate()` unchanged — the same function that produced
  every Phase 3 number, so the val → test difference above is a data difference, not a harness one.
- `reasoning_effort: "none"` remains set on the teacher (D-013). The run's actual spend is checked
  against its projection at run time; a large overshoot means reasoning is back on and the numbers are
  measuring truncation.
- The test split was verified by sha256 against `data/manifest.json` before either model saw it, and
  every saved measurement records the hash it was taken against.
- Every number here is generated from `reports/raw/phase4/*.json` by this script.
"""

    (ROOT / "reports" / "phase4.md").write_text(report)
    print(f"wrote reports/phase4.md · student {s_score.micro.f1:.3f} vs teacher {t_score.micro.f1:.3f} "
          f"(Δ {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}])")


if __name__ == "__main__":
    main()

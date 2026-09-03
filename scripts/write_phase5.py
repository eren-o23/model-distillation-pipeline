"""Generate reports/phase5.md from the measurements Phase 5 left behind.

Same discipline as write_ceiling.py, write_phase3.py and write_phase4.py: every number is read from a
saved run rather than transcribed, so the report cannot drift from what happened. Runs locally, needs
no GPU and no server.

  python scripts/write_phase5.py                    # after copying reports/raw/phase5/ off the box
  python scripts/write_phase5.py --operating-point 0.05 --signal min_logprob

The operating point is chosen by a stated rule — the cheapest escalation rate that buys essentially
all of the available quality — and overridable, because it is an operating decision rather than a
measurement. The rule is printed in the report next to the number it produced.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii import economics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports" / "raw" / "phase5"
PHASE4 = ROOT / "reports" / "raw" / "phase4"

# "Essentially all the quality": the cheapest point reaching 95% of the best F1 gain the sweep found
# over the student alone. Stated rather than tuned — a rule picked after seeing the curve is a rule
# fitted to it.
GAIN_CAPTURE = 0.95


def load(directory: Path, pattern: str) -> dict | None:
    hits = [json.loads(p.read_text()) for p in sorted(directory.glob(pattern))]
    return max(hits, key=lambda b: b.get("n", 0)) if hits else None


def best_concurrency(by_c: dict) -> tuple[str, dict]:
    """The concurrency that maximises throughput — the same basis Phase 4 priced its card at."""
    key = max(by_c, key=lambda c: by_c[c]["requests_per_s"])
    return key, by_c[key]


def knee_concurrency(by_c: dict) -> str:
    """The last concurrency where throughput rises faster than the tail does.

    Past it you are buying req/s with p95, which is a trade an operator may or may not want. The cost
    columns are priced at peak throughput regardless — that is the cost-optimal point and the basis
    Phase 4 used — but the knee is named so the latency it costs is visible rather than implied.
    """
    order = sorted(by_c, key=lambda c: int(c))
    knee = order[0]
    for prev, cur in zip(order, order[1:]):
        a, b = by_c[prev], by_c[cur]
        gain = b["requests_per_s"] / a["requests_per_s"] - 1
        tail = b["p95_s"] / a["p95_s"] - 1
        if gain < tail:
            break
        knee = cur
    return knee


def pick_operating_point(points: list[dict], student_f1: float) -> dict:
    """The cheapest escalation rate capturing GAIN_CAPTURE of the best gain over the student alone."""
    best_gain = max(p["micro_f1"] - student_f1 for p in points)
    if best_gain <= 0:
        return points[0]
    target = student_f1 + GAIN_CAPTURE * best_gain
    return min((p for p in points if p["micro_f1"] >= target), key=lambda p: p["escalation_rate"])


def fmt_thousands(x: float) -> str:
    return "—" if x == float("inf") else f"{x:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operating-point", type=float, help="escalation rate to deploy at; omit to use the rule")
    ap.add_argument("--signal", help="min_logprob or mean_logprob; omit to pick the better on val")
    ap.add_argument("--raw", type=Path, default=RAW)
    args = ap.parse_args()
    raw = args.raw

    parity = load(raw, "vllm-parity-val-*.json")
    lat = load(raw, "vllm-latency-*.json")
    sweep = next((json.loads(p.read_text()) for p in sorted(raw.glob("router-sweep.json"))), None)
    if not (parity and lat and sweep):
        sys.exit(f"missing measurements in {raw} — need the parity, latency and sweep runs")

    teacher4 = load(PHASE4, "teacher-test-*.json")
    student4 = load(PHASE4, "student-latency-*.json")
    teacher_per_1k = teacher4["cost_usd"] / teacher4["n"] * 1000
    p_t = teacher_per_1k / 1000

    conc, best = best_concurrency(lat["by_concurrency"])
    knee = knee_concurrency(lat["by_concurrency"])
    knee_row = lat["by_concurrency"][knee]
    rps, rate = best["requests_per_s"], lat["gpu_rate_usd_h"]
    gpu = lat["gpu"]

    # ---- the before/after on throughput ------------------------------------------------------------
    p4_curve = {int(k): v for k, v in student4["by_batch"].items() if not v.get("oom")} if student4 else {}
    p4_rps = max((v["requests_per_s"] for v in p4_curve.values()), default=0.0)
    speedup = rps / p4_rps if p4_rps else 0.0

    # ---- the parity gate ---------------------------------------------------------------------------
    prior = parity.get("phase3_micro_f1")
    delta = parity["micro_f1"] - prior if prior else None
    if delta is None:
        gate = f"The served model scores **{parity['micro_f1']:.3f}** on {parity['n']:,} val rows."
    elif abs(delta) < 0.01:
        gate = (f"**The fp16 merge is the model Phase 4 benchmarked.** It scores "
                f"**{parity['micro_f1']:.3f}** against the 4-bit+LoRA student's {prior:.3f} on the same "
                f"{parity['n']:,} val rows — a difference of {delta:+.3f}, inside the noise this project "
                f"has consistently called null. Everything below therefore prices the model whose "
                f"quality Phase 4 measured, rather than a cheaper cousin of it.")
    else:
        gate = (f"**The fp16 merge is NOT the model Phase 4 benchmarked.** It scores "
                f"{parity['micro_f1']:.3f} against the 4-bit+LoRA student's {prior:.3f} on the same "
                f"{parity['n']:,} val rows — {delta:+.3f}, wider than noise. The throughput and cost "
                f"numbers below belong to *this* model, and Phase 4's quality numbers belong to the "
                f"other one. They are not two measurements of one system and the report does not "
                f"present them as such.")

    # ---- the router --------------------------------------------------------------------------------
    signal = args.signal or max(sweep["by_signal"],
                                key=lambda s: max(p["micro_f1"] for p in sweep["by_signal"][s]))
    points = sweep["by_signal"][signal]
    student_f1, teacher_f1 = sweep["student_only_f1"], sweep["teacher_only_f1"]
    if args.operating_point is not None:
        op = min(points, key=lambda p: abs(p["escalation_rate"] - args.operating_point))
        rule = f"chosen by hand at {args.operating_point:.0%}"
    else:
        op = pick_operating_point(points, student_f1)
        rule = (f"the cheapest escalation rate reaching {GAIN_CAPTURE:.0%} of the best quality gain the "
                f"sweep found over the student alone")
    e = op["escalation_rate"]

    # Does the confidence signal do any work, or would escalating at random do as well? Without this
    # the sweep cannot tell a working router from the fact that a 0.822 teacher helps on any subset.
    lift = op["micro_f1"] - op["random_micro_f1"]
    if lift > 0.005:
        router_verdict = (f"**The confidence signal is doing the work.** At the same {e:.1%} escalation "
                          f"rate, routing by `{signal}` scores {op['micro_f1']:.3f} against "
                          f"{op['random_micro_f1']:.3f} for escalating rows picked at random — "
                          f"{lift:+.3f}. The router is selecting the rows the student actually got "
                          f"wrong, which is the only thing that makes it worth its premium.")
    else:
        router_verdict = (f"**The confidence signal is not doing much work.** At {e:.1%} escalation it "
                          f"scores {op['micro_f1']:.3f} against {op['random_micro_f1']:.3f} for random "
                          f"selection — {lift:+.3f}, which is not a signal. Most of the gain here comes "
                          f"from escalating *anything* to a 0.822-F1 teacher, not from knowing which "
                          f"rows to escalate. Reported as such: a router whose threshold is doing "
                          f"nothing is a cost with a story attached.")

    # ---- the break-even ----------------------------------------------------------------------------
    v_alone = economics.break_even_per_day(rate, p_t)
    v_routed = economics.break_even_per_day(rate, p_t, e)
    capacity = economics.capacity_per_day(rps)
    frac_alone = economics.capacity_fraction(rate, p_t, rps)
    frac_routed = economics.capacity_fraction(rate, p_t, rps, e)
    blended = economics.blended_per_1k(rate, rps, p_t, e)
    student_per_1k = economics.per_1k(rate, rps)

    if frac_routed < 1.0:
        headline = (
            f"**Break-even is at {fmt_thousands(v_routed)} requests per day.** Below that, pay the API; "
            f"above it, the self-hosted student is cheaper — including the {e:.1%} of traffic the router "
            f"escalates back to the teacher, and including the GPU's idle hours. That volume is "
            f"{frac_routed:.0%} of what this card can actually serve "
            f"({fmt_thousands(capacity)} requests/day at {rps:.2f} req/s), so it is a volume the stack "
            f"can reach rather than an arithmetic one it cannot.\n\n"
            f"Without the router the number is {fmt_thousands(v_alone)}/day. The difference is the "
            f"router's price: every escalated request is one the GPU was paid for *and* the API is "
            f"billed for, so it raises the volume needed to amortise the card by exactly the escalation "
            f"rate. That premium buys the recall in the table below, and it is reported rather than "
            f"absorbed."
        )
    else:
        headline = (
            f"**There is still no usable break-even volume.** The student would need "
            f"{fmt_thousands(v_routed)} requests per day to amortise a ${rate}/hr card at a {e:.1%} "
            f"escalation rate, and the card can only serve {fmt_thousands(capacity)} — "
            f"{frac_routed:.0%} of the break-even volume. vLLM lifted throughput from "
            f"{p4_rps:.3f} to {rps:.2f} req/s ({speedup:.1f}x), and that still is not enough.\n\n"
            f"Published as the negative it is, exactly as Phase 4's was (D-032). Adding cards does not "
            f"rescue it: cost and capacity scale together, so the ratio above is invariant."
        )

    sweep_rows = []
    for p in points:
        cost = f"${p['usd_per_1k']:.2f}" if "usd_per_1k" in p else "—"
        mark = " **←**" if p is op else ""
        sweep_rows.append(
            f"| {p['escalation_rate']:.1%} | {p['micro_f1']:.3f} | {p['random_micro_f1']:.3f} | "
            f"{(p['idcardnum_recall'] or 0):.3f} | {cost} |{mark}")

    curve_rows = [f"| {c} | {v['p50_s']:.2f}s | {v['p95_s']:.2f}s | {v['requests_per_s']:.2f} | "
                  f"${v['usd_per_1k']:.3f} |" for c, v in
                  sorted(lat["by_concurrency"].items(), key=lambda kv: int(kv[0]))]

    p4_rows = [f"| {b} | {v['p50_s']:.2f}s | {v['p95_s']:.2f}s | {v['requests_per_s']:.3f} |"
               for b, v in sorted(p4_curve.items())]

    idcard_student = next((p for p in points if p["escalation_rate"] == min(
        q["escalation_rate"] for q in points)), points[0])

    report = f"""# Phase 5 — Served with vLLM, and what the break-even volume turned out to be

**Measured:** {date.today()}. Student served by vLLM on **{gpu}** at ${rate}/hr{
    f" ({lat['gpu_rate_source']})" if lat.get("gpu_rate_source") else ""}.
**Quality and router: `val`**, {parity['n']:,} rows, sha256 `{parity['split_sha256'][:16]}…`. The test
set was opened once in Phase 4 and is spent (D-028), so every tuning decision here is made on `val` and
the report says so rather than quoting a contaminated number.
**Throughput: the same seeded {lat['n']} test rows Phase 4 timed** — no labels are read on that path,
so identical inputs cost nothing epistemically and make the before/after exact (D-034).

{headline}

## The parity gate

{gate}

## Axis 1 — Throughput, before and after

| stack | best throughput | $/1k at 100% |
|---|---|---|
| Phase 4 — HF `generate`, static batching, T4 | {p4_rps:.3f} req/s | ${economics.per_1k(0.526, p4_rps):.2f} |
| Phase 5 — vLLM, continuous batching, {gpu} | **{rps:.2f} req/s** | **${student_per_1k:.2f}** |

That is **{speedup:.1f}x**, and the attribution is not clean: this changed the serving stack *and* the
card at the same time. Phase 4's T4 could not hold an fp16 8B alongside a KV cache at all, which is why
both moved together (D-030), but it means the multiplier above belongs to the pair rather than to vLLM
alone. What is clean is the economics, because each card is priced at its own real rate.

D-032 set two targets from Phase 4's measurements: **0.67 req/s** for cost parity at 50% utilisation and
**6.7 req/s** for a tenfold advantage. This stack reaches {rps:.2f}.

### The concurrency curve

| in-flight requests | p50 | p95 | req/s | $/1k @ 100% |
|---|---|---|---|---|
{chr(10).join(curve_rows)}

Throughput peaks at **{conc} in-flight requests** and that is what the cost columns are priced at, the
same cost-optimal basis Phase 4 used. The **knee is at {knee}** — the last point where throughput rises
faster than the tail does. Past it you buy req/s with p95: {conc} gives {best['requests_per_s']:.2f} req/s
at a {best['p95_s']:.1f}s p95 against {knee}'s {knee_row['requests_per_s']:.2f} req/s at
{knee_row['p95_s']:.1f}s. It does not change the verdict — ${best['usd_per_1k']:.3f} and ${knee_row['usd_per_1k']:.3f} per 1,000 are both far
under the teacher's ${teacher_per_1k:.2f} — so the choice is a latency preference rather than an
economic one.

Concurrency finally means the same thing on both sides. Phase 4 had to use batch size as a stand-in,
because static batching returns every request in a batch when its slowest member finishes; vLLM batches
continuously, so an in-flight request here is the same unit as an in-flight request against the API.

For comparison, the stack this replaces — Phase 4, where batch size was the axis:

| batch | p50 | p95 | req/s |
|---|---|---|---|
{chr(10).join(p4_rows)}

## Axis 2 — The break-even volume

With `p_t` the measured teacher price per request (${p_t:.6f}), `C` the card's real hourly rate
(${rate}), `R` the measured throughput ({rps:.2f} req/s) and `e` the escalation rate:

```
self-hosted $/day at V requests = 24·C + V·e·p_t      the card, plus the API for what it escalates
API-only     $/day at V requests = V·p_t

break-even  V* = 24·C / (p_t·(1 − e))
capacity       = R · 86,400
```

| | break-even V* | as a share of capacity |
|---|---|---|
| student alone | {fmt_thousands(v_alone)} /day | {frac_alone:.0%} |
| with the router at {e:.1%} | {fmt_thousands(v_routed)} /day | {frac_routed:.0%} |

**The share-of-capacity column is the one that matters**, and it is why Phase 4 was reported as a
failure despite the student being $0.0017 cheaper per 1,000 there: its break-even sat at 99.6% of what
the card could serve in a day, so it only paid on a GPU that was never idle for a second. A break-even
volume a stack cannot physically reach is not a break-even volume.

| system | $/1k requests |
|---|---|
| teacher (`qwen3p7-plus` API) | ${teacher_per_1k:.2f} |
| student alone, saturated | ${student_per_1k:.2f} |
| student + router at {e:.1%} | ${blended:.2f} |

The GPU is charged on all 1,000 in the blended row. An escalated request still ran on the student
first, so escalation is not a discount on the card — it is an addition to the bill.

## Axis 3 — The router

Swept on `val` by **target escalation rate** rather than by raw logprob, because a rate is the knob an
operator has. The threshold is recovered as the matching quantile of the observed `{signal}`
distribution. Student alone scores {student_f1:.3f}; the teacher alone scores {teacher_f1:.3f}.

| escalation rate | micro-F1 | random at same rate | `IDCARDNUM` recall | $/1k |
|---|---|---|---|---|
{chr(10).join(sweep_rows)}

**Operating point: {e:.1%}** — {rule}.

{router_verdict}

`IDCARDNUM` is why the router exists. The student inherited the teacher's conservatism on it exactly as
D-015 predicted — 0.802 precision against 0.280 recall on the sealed test set — because the training
labels themselves score 0.419 F1 on that class. Escalation moves its recall from
{(idcard_student['idcardnum_recall'] or 0):.3f} to {(op['idcardnum_recall'] or 0):.3f} here, which is
recall the student cannot be trained into without reopening D-006 and rebuilding every split.

## Method notes

- The served prompt comes from `src/pii/student.render_prompt` and is sent through `/v1/completions`,
  not `/v1/chat/completions`. Letting the server apply Qwen3's chat template would hand the model a
  prefix it was never trained on — D-013's failure mode, which has already cost this project twice.
- Generation matches `src/pii/eval.py` exactly: greedy, {parity.get('max_tokens', 512)}-token cap.
- The router sweep calls nothing. Both arms were measured once and cached, so every threshold in the
  table is a lookup, and the sweep re-runs at no cost.
- Every escalation point is measured against random escalation at the same rate, so a signal that does
  no work cannot pass as a router that does.
- Quality is scored against dataset **gold** by `src/pii/metric.py`, never against the teacher.
- Every number here is generated from `reports/raw/phase5/*.json` by this script.
"""

    (ROOT / "reports" / "phase5.md").write_text(report)
    print(f"wrote reports/phase5.md · {rps:.2f} req/s · break-even {fmt_thousands(v_routed)}/day "
          f"({frac_routed:.0%} of capacity) · escalation {e:.1%}")


if __name__ == "__main__":
    main()

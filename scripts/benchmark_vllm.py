"""Phase 5: the student under vLLM — the parity gate, then throughput at rising concurrency.

Two passes, both free once the box is rented.

  python scripts/benchmark_vllm.py --parity  --gpu "NVIDIA L4" --gpu-rate 0.75
  python scripts/benchmark_vllm.py --latency --gpu "NVIDIA L4" --gpu-rate 0.75

**The parity pass is a gate, not a formality.** Phase 4 measured a 4-bit NF4 base with a LoRA adapter
on top; this serves an fp16 merge of the two. That is a different model, and pricing the throughput of
a model whose quality was never re-measured is precisely the flattery this project has refused four
times. It re-scores all 1,000 `val` rows and compares against Phase 3's 0.833 on those same rows.

It also does double duty: the per-row logprobs it saves are what `benchmark_router.py` sweeps offline,
so the escalation curve costs no GPU time of its own.

The latency pass runs on the **same seeded 100 test rows Phase 4 timed** — see D-034. A latency
measurement reads no gold labels and selects nothing, so identical inputs cost nothing epistemically
and make the before/after against Phase 4's curve exact rather than approximate.

Writes reports/raw/phase5/*.json. Copy them off the box and push before the session ends: on a metered
machine, losing a measurement costs money and not just the 45 minutes Phase 3 paid.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import latency_sample, load_split, verify_frozen  # noqa: E402
from src.pii.economics import per_1k  # noqa: E402
from src.pii.metric import Score  # noqa: E402
from src.pii.router import client, complete, tokenizer  # noqa: E402
from src.pii.student import SHORT_SYSTEM, render_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports" / "raw" / "phase5"
PHASE3_FULL_VAL = ROOT / "reports" / "raw" / "phase3" / "adapter-pii-qwen3-8b-lora-r8-full.json"

# Client-side concurrency is the real axis here, unlike Phase 4 where batch size stood in for it.
# vLLM batches continuously server-side, so in-flight requests mean the same thing they mean for the
# teacher's API — which is what finally makes the two latency columns comparable rather than analogous.
CONCURRENCY = (1, 8, 16, 32, 64, 128)
WARMUP = 5


def run_all(cli, model: str, prompts: list[str], workers: int) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: complete(cli, model, p), prompts))


def parity(cli, model: str, tok, rows: list[dict], split_sha: str, workers: int) -> dict:
    """Score every val row through the served model. The number that gates the rest of the phase."""
    prompts = [render_prompt(tok, r["source_text"], SHORT_SYSTEM) for r in rows]
    # Logged once so the served prompt can be diffed by eye against what training rendered. A silent
    # mismatch here does not raise — it just looks like a bad fine-tune.
    print(f"prompt[0] as sent, {len(prompts[0])} chars:\n---\n{prompts[0]}\n---", flush=True)

    t0 = time.perf_counter()
    comps = run_all(cli, model, prompts, workers)
    seconds = time.perf_counter() - t0

    errors = [(r["uid"], c.error) for r, c in zip(rows, comps, strict=True) if c.error]
    s = Score()
    for row, c in zip(rows, comps, strict=True):
        s.add(row["entities"], c.entities, row["source_text"])
    print(s.table(), flush=True)
    if errors:
        # Reported separately from schema-invalid, which counts only answers the model actually
        # returned malformed. A request that never reached it is not a quality defect.
        print(f"\n  {len(errors)} of {len(rows)} requests never reached the model — "
              f"NOT counted as schema-invalid. First: {errors[0][1]}", flush=True)

    prior = json.loads(PHASE3_FULL_VAL.read_text())["micro_f1"] if PHASE3_FULL_VAL.exists() else None
    if prior is not None:
        print(f"\nparity gate: vLLM fp16 merge {s.micro.f1:.3f} against Phase 3's 4-bit+LoRA "
              f"{prior:.3f} on these same {len(rows)} val rows — delta {s.micro.f1 - prior:+.3f}",
              flush=True)

    return {
        "tag": f"vllm-parity-val-{len(rows)}",
        "split": "val",
        "split_sha256": split_sha,
        "date": str(date.today()),
        "model": model,
        "n": len(rows),
        "workers": workers,
        "wall_seconds": round(seconds, 1),
        "micro_f1": s.micro.f1,
        "micro_precision": s.micro.precision,
        "micro_recall": s.micro.recall,
        "schema_invalid": s.schema_invalid,
        "hallucinated": s.hallucinated,
        "transport_errors": errors,
        "phase3_micro_f1": prior,
        "uids": [r["uid"] for r in rows],
        # The router sweep's entire student side. Saved per row so every threshold is arithmetic.
        "rows": [
            {"uid": r["uid"], "entities": c.entities, "mean_logprob": c.mean_logprob,
             "min_logprob": c.min_logprob, "n_tokens": c.n_tokens, "error": c.error}
            for r, c in zip(rows, comps, strict=True)
        ],
    }


def latency(cli, model: str, tok, rows: list[dict], split_sha: str, rate: float) -> dict:
    """Throughput and tail latency as concurrency rises, up to where the tail stops paying for it."""
    prompts = [render_prompt(tok, r["source_text"], SHORT_SYSTEM) for r in rows]

    print(f"warming up ({WARMUP} requests, discarded) …", flush=True)
    run_all(cli, model, prompts[:WARMUP], WARMUP)

    out = {}
    for c in CONCURRENCY:
        t0 = time.perf_counter()
        comps = run_all(cli, model, prompts, c)
        total = time.perf_counter() - t0

        lat = sorted(x.seconds for x in comps)
        rps = len(prompts) / total
        out[str(c)] = {
            "p50_s": median(lat),
            "p95_s": lat[min(int(0.95 * len(lat)), len(lat) - 1)],
            "requests_per_s": rps,
            "total_s": round(total, 1),
            "usd_per_1k": per_1k(rate, rps),
            "schema_invalid": sum(1 for x in comps if x.entities is None),
        }
        r = out[str(c)]
        print(f"  concurrency {c:>3}: p50 {r['p50_s']:.2f}s  p95 {r['p95_s']:.2f}s  "
              f"{rps:.2f} req/s  ${r['usd_per_1k']:.3f}/1k", flush=True)

    return {
        "tag": f"vllm-latency-{len(rows)}",
        "split": "test",
        "split_sha256": split_sha,
        "date": str(date.today()),
        "model": model,
        "n": len(rows),
        "by_concurrency": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name as vLLM serves it (see /v1/models)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--parity", action="store_true", help="the val quality gate")
    ap.add_argument("--latency", action="store_true", help="the concurrency sweep")
    ap.add_argument("--parity-workers", type=int, default=64)
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--limit", type=int, help="cap rows, for a smoke test")
    # D-029's revisit clause: on rented hardware the real invoice replaces Phase 4's quoted T4 rate.
    # Recorded per run rather than as a module constant, because it changes with the box.
    ap.add_argument("--gpu", required=True, help='the card, e.g. "NVIDIA L4"')
    ap.add_argument("--gpu-rate", type=float, required=True, help="its real $/hr, from the invoice")
    ap.add_argument("--gpu-source", default="", help="where that rate came from, for the report")
    args = ap.parse_args()

    if not (args.parity or args.latency):
        sys.exit("pick one: --parity (the gate) or --latency (the sweep)")

    cli = client(args.base_url)
    tok = tokenizer()

    if args.parity:
        sha = verify_frozen("val")
        rows = load_split("val")
        print(f"val split verified: {len(rows)} rows, sha256 {sha[:16]}… matches the manifest")
        payload = parity(cli, args.model, tok, rows[: args.limit] if args.limit else rows, sha,
                         args.parity_workers)
    else:
        # Timing only. Gold labels are never read on this path — see D-034.
        sha = verify_frozen("test")
        rows = latency_sample(load_split("test", allow_test=True), args.latency_n)
        print(f"timing on the same seeded {len(rows)} test rows Phase 4 used (no labels read)")
        payload = latency(cli, args.model, tok, rows, sha, args.gpu_rate)

    payload |= {"gpu": args.gpu, "gpu_rate_usd_h": args.gpu_rate, "gpu_rate_source": args.gpu_source}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{payload['tag']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote reports/raw/phase5/{path.name}", flush=True)


if __name__ == "__main__":
    main()

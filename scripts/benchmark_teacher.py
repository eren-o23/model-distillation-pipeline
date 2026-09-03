"""Phase 4: the teacher on the sealed test set — quality, cost and latency in one paid run.

This is the only run in Phase 4 that spends money, and the first code in the project permitted to pass
`allow_test=True`. It opens the test split once, scores the teacher on all 1,000 rows at the same
concurrency Phase 2 used, and times every call on the way past so the latency axis costs nothing extra.

  python scripts/benchmark_teacher.py                    # dry run: prints the estimate, spends nothing
  python scripts/benchmark_teacher.py --limit 20 --yes   # smoke test, ~$0.01
  python scripts/benchmark_teacher.py --yes              # the real run, ~$0.50

Raw predictions, usage and per-call latencies land in reports/raw/phase4/. scripts/write_phase4.py turns
those into the report, so no number in it is typed by hand — and re-running this script reuses the
checkpoint rather than re-spending.
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
from src.pii.metric import Score  # noqa: E402
from src.pii.teacher import Usage, client, extract, is_api_failure, price_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Phase 4 measured the teacher on `test`; Phase 5 needs it on `val`, because the test set is spent
# (D-028) and the router must be tuned somewhere. Same script, same checkpointing, same cost gate —
# a second paid script would be a second place to get the retry logic subtly wrong.
RAW_BY_SPLIT = {"test": ROOT / "reports" / "raw" / "phase4", "val": ROOT / "reports" / "raw" / "phase5"}
RAW = RAW_BY_SPLIT["test"]
# Phase 1's ceiling run, kept as the token profile this phase's cost estimate is built from. Measuring
# 200 real calls beats the chars/4 heuristic: it already knows what this prompt and this model cost.
CEILING_CHECKPOINT = ROOT / "reports" / "raw" / "qwen3p7-plus-200.json"
MODEL = "accounts/fireworks/models/qwen3p7-plus"

# 8, not 16. Phase 2's first full run at 16 workers took sustained 429s on 684 of 7,992 rows; at 8 the
# retry pass recovered every one with zero errors. The run is bound by the serverless rate limit rather
# than by local concurrency, so more threads buy nothing and cost measurement accuracy.
WORKERS = 8


def token_profile() -> tuple[float, float]:
    """Measured input/output tokens per example, from the Phase 1 ceiling checkpoint."""
    blob = json.loads(CEILING_CHECKPOINT.read_text())
    usage, n = Usage(**blob["usage"]), len(blob["predictions"])
    return usage.prompt_tokens / n, usage.completion_tokens / n


def estimate(n: int, model_id: str) -> float:
    per_in, per_out = token_profile()
    pin, pout = price_for(model_id)
    return (per_in * pin + per_out * pout) * n / 1_000_000


def timed(cli, model_id: str, row: dict) -> dict:
    """One call, with its own wall clock. The latency axis is a by-product of the quality run."""
    t0 = time.perf_counter()
    entities, usage = extract(cli, model_id, row["source_text"])
    return {
        "uid": row["uid"],
        "entities": entities,
        "in": usage.prompt_tokens,
        "out": usage.completion_tokens,
        "seconds": round(time.perf_counter() - t0, 3),
    }


def fetch(cli, model_id: str, rows: list[dict], workers: int) -> list[dict]:
    """Run `rows` at `workers` concurrency, then retry anything that never reached the model.

    A 429 or a timeout returns None with nothing billed, exactly like an unparseable answer does. Left
    conflated, those rows would be published as teacher schema-invalid — a quality defect that did not
    happen — so they are retried instead, and whatever survives is reported separately.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda r: timed(cli, model_id, r), rows))

    by_uid = {r["uid"]: r for r in records}
    failed = [r for r in rows if is_api_failure(by_uid[r["uid"]]["entities"], by_uid[r["uid"]]["in"])]
    if failed:
        print(f"  {len(failed)} transport failures (nothing billed) — retrying at {workers // 2}", flush=True)
        with ThreadPoolExecutor(max_workers=max(workers // 2, 1)) as pool:
            for rec in pool.map(lambda r: timed(cli, model_id, r), failed):
                by_uid[rec["uid"]] = rec
    return [by_uid[r["uid"]] for r in rows]


def run(cli, model_id: str, rows: list[dict], workers: int, tag: str, split_sha: str,
        split: str = "test") -> dict:
    """Fetch (or reuse a checkpoint) and persist before scoring. Never re-spends silently."""
    path = RAW / f"{tag}.json"
    if path.exists():
        print(f"  REUSING CHECKPOINT {path.name} — no API calls, no spend.", flush=True)
        print(f"  Delete it to force a genuine re-run: rm {path.relative_to(ROOT)}", flush=True)
        return json.loads(path.read_text())

    records = fetch(cli, model_id, rows, workers)
    usage = Usage(
        prompt_tokens=sum(r["in"] for r in records),
        completion_tokens=sum(r["out"] for r in records),
    )
    payload = {
        "tag": tag,
        "model": model_id,
        "split": split,
        "split_sha256": split_sha,
        "n": len(records),
        "workers": workers,
        "date": str(date.today()),
        "uids": [r["uid"] for r in records],
        "predictions": [r["entities"] for r in records],
        "latency_s": [r["seconds"] for r in records],
        # Rows that never reached the model, after the retry above. Kept as uids so the report can
        # subtract them from the schema-invalid count instead of publishing them as teacher defects.
        "api_failures": [r["uid"] for r in records if is_api_failure(r["entities"], r["in"])],
        "usage": vars(usage),
        "cost_usd": usage.cost(model_id),
    }
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  checkpointed -> {path.relative_to(ROOT)}", flush=True)
    return payload


def summarise(payload: dict, rows: list[dict]) -> None:
    s = Score()
    for row, pred in zip(rows, payload["predictions"], strict=True):
        s.add(row["entities"], pred, row["source_text"])
    print(s.table(), flush=True)

    failures = len(payload["api_failures"])
    if failures:
        print(f"  NOTE: {failures} of those {s.schema_invalid} unusable rows never reached the model "
              "(unbilled transport failures, not teacher defects).", flush=True)

    lat = sorted(payload["latency_s"])
    p95 = lat[min(int(0.95 * len(lat)), len(lat) - 1)]
    print(f"\n  cost ${payload['cost_usd']:.4f}  ·  {payload['usage']['prompt_tokens']:,} in / "
          f"{payload['usage']['completion_tokens']:,} out", flush=True)
    print(f"  latency @{payload['workers']} concurrent: p50 {median(lat):.2f}s  p95 {p95:.2f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", default="test", choices=sorted(RAW_BY_SPLIT),
                    help="test = the Phase 4 benchmark; val = the Phase 5 router's teacher arm")
    ap.add_argument("--limit", type=int, help="cap rows, for the smoke test; omit for the full 1,000")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--serial-n", type=int, default=50,
                    help="rows for the concurrency-1 latency pass; 0 skips it")
    ap.add_argument("--yes", action="store_true", help="required to actually spend credits")
    args = ap.parse_args()

    global RAW
    RAW = RAW_BY_SPLIT[args.split]

    # The seal, checked before anything else: this must be the file Phase 1 froze, not one rebuilt since.
    split_sha = verify_frozen(args.split)
    rows = load_split(args.split, allow_test=args.split == "test")
    assert len(rows) == 1000, f"{args.split} split is {len(rows)} rows, expected 1000"
    print(f"{args.split} split verified: {len(rows)} rows, sha256 {split_sha[:16]}… matches the manifest")

    # The concurrency-1 pass runs on a seeded subset of the WHOLE split, so the student times itself on
    # identical rows later. Under --limit it is trimmed to what the smoke test actually fetched.
    serial_rows = latency_sample(rows, args.serial_n) if args.serial_n else []
    if args.limit:
        rows = rows[: args.limit]
        serial_rows = serial_rows[: min(args.serial_n, args.limit)]

    cost = estimate(len(rows), args.model) + estimate(len(serial_rows), args.model)
    per_in, per_out = token_profile()
    print(f"\n{args.model}")
    print(f"  quality + concurrency-{args.workers} latency: {len(rows)} rows")
    print(f"  concurrency-1 latency:                  {len(serial_rows)} rows")
    print(f"  at {per_in:.0f} in / {per_out:.0f} out tokens per example (measured in Phase 1)")
    print(f"  estimated total: ~${cost:.2f}")

    if not args.yes:
        print("\nDry run. Nothing was called and nothing was spent. Re-run with --yes.")
        return

    cli = client()
    print(f"\nquality pass — {len(rows)} rows at {args.workers} concurrent …", flush=True)
    main_run = run(cli, args.model, rows, args.workers, f"teacher-{args.split}-{len(rows)}",
                   split_sha, args.split)
    summarise(main_run, rows)

    # Phase 5 skips this: the teacher's serial latency was already measured in Phase 4 and has not
    # changed. Re-running it would spend money to reproduce a number already in the report.
    spent = main_run["cost_usd"]
    if serial_rows:
        print(f"\nlatency pass — {len(serial_rows)} rows, serial …", flush=True)
        serial = run(cli, args.model, serial_rows, 1,
                     f"teacher-latency-serial-{len(serial_rows)}", split_sha, args.split)
        lat = sorted(serial["latency_s"])
        p95 = lat[min(int(0.95 * len(lat)), len(lat) - 1)]
        print(f"  latency @1 concurrent: p50 {median(lat):.2f}s  p95 {p95:.2f}s", flush=True)
        spent += serial["cost_usd"]
    print(f"\nspent ${spent:.4f} against a ~${cost:.2f} estimate")
    if spent > cost * 1.5:
        print("  OVERSHOT: a >1.5x overshoot means the teacher is thinking again (D-013) — "
              "check reasoning_effort before trusting these numbers.")


if __name__ == "__main__":
    main()

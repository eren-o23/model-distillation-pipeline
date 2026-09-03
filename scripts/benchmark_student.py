"""Phase 4: the student on the sealed test set — quality, latency and throughput. Runs on Kaggle's T4.

Two passes, both free. The quality pass calls `src.pii.eval.evaluate()` exactly as Phase 3 did, so the
test number and the val number are produced by one harness and their difference means something. The
latency pass times the same generation path at four batch sizes over a seeded sample shared with the
teacher, which is what makes "comparable concurrency" comparable.

  python scripts/benchmark_student.py --adapter erenrosman/pii-qwen3-8b-lora-r8
  python scripts/benchmark_student.py --adapter erenrosman/pii-qwen3-8b-lora-r16
  python scripts/benchmark_student.py --adapter erenrosman/pii-qwen3-8b-lora-r8 --latency

Writes reports/raw/phase4/*.json. Download them before the session ends — /kaggle/working does not
reliably survive, and Phase 3 lost a full eval that way.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Nothing here uses Trainer, but the same rule applies for a different reason: with both T4s visible the
# memory figures below would be reported against a card the model is not on. Pin to one.
if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.pii.data import latency_sample, load_split, verify_frozen  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "reports" / "raw" / "phase4"

# 1 is the single-request case a user actually experiences; 8 matches the teacher's concurrency; 16 is
# what Phase 3 evaluated at; 32 is there to find the ceiling. Static batching means every request in a
# batch is returned together, so batch size is this stack's analogue of concurrency — and the point at
# which it OOMs or stops paying is the finding, not a failure of the run.
BATCH_SIZES = (1, 8, 16, 32)


def tag_for(adapter: str, n: int) -> str:
    return f"student-{adapter.rstrip('/').split('/')[-1]}-test-{n}"


def quality(model, tok, rows: list[dict], adapter: str, split_sha: str, batch_size: int) -> dict:
    """The headline number. `evaluate()` unchanged — a re-implementation would measure the harness."""
    from src.pii.eval import as_dict, evaluate

    t0 = time.perf_counter()
    score, samples = evaluate(model, tok, rows, batch_size=batch_size, n_samples=None)
    seconds = time.perf_counter() - t0

    payload = as_dict(score) | {
        "tag": tag_for(adapter, len(rows)),
        "adapter": adapter,
        "split": "test",
        "split_sha256": split_sha,
        "date": str(date.today()),
        "batch_size": batch_size,
        "eval_seconds": round(seconds, 1),
        "uids": [r["uid"] for r in rows],
    }
    print(f"\n[{payload['tag']}] micro-F1 {score.micro.f1:.3f}  P {score.micro.precision:.3f}  "
          f"R {score.micro.recall:.3f}  schema-invalid {score.schema_invalid}/{score.n_examples}  "
          f"({seconds:.0f}s)", flush=True)
    print(score.table(), flush=True)
    # samples is every row here, which is what write_phase4.py bootstraps the confidence interval from.
    return payload | {"samples": samples}


def latency(model, tok, rows: list[dict], adapter: str, split_sha: str) -> dict:
    """Time the same generation path at each batch size, on rows the teacher was also timed on.

    Batches are taken in the sample's own order rather than length-sorted. Sorting is right for a bulk
    eval — it is most of its throughput — but it would quietly report the latency of a workload nobody
    runs, where every request arrives alongside others of its own length.
    """
    import torch
    from src.pii.eval import generate
    from src.pii.student import render_prompt

    prompts = [render_prompt(tok, r["source_text"]) for r in rows]
    out = {}
    for b in BATCH_SIZES:
        torch.cuda.reset_peak_memory_stats()
        per_request, total_s = [], 0.0
        try:
            for start in range(0, len(prompts), b):
                chunk = prompts[start : start + b]
                t0 = time.perf_counter()
                # batch_size=len(chunk) so this is exactly one batch: the internal length sort cannot
                # regroup anything, and the wall clock covers precisely the requests being timed.
                generate(model, tok, chunk, batch_size=len(chunk))
                dt = time.perf_counter() - t0
                # Every request in a static batch is returned when the slowest one finishes, so that is
                # the latency each of them observes.
                per_request += [round(dt, 3)] * len(chunk)
                total_s += dt
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out[str(b)] = {"oom": True}
            print(f"  batch {b}: OOM — the T4's ceiling, recorded rather than worked around", flush=True)
            continue

        lat = sorted(per_request)
        out[str(b)] = {
            "latency_s": per_request,
            "p50_s": median(lat),
            "p95_s": lat[min(int(0.95 * len(lat)), len(lat) - 1)],
            "requests_per_s": len(prompts) / total_s,
            "total_s": round(total_s, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        r = out[str(b)]
        print(f"  batch {b:>2}: p50 {r['p50_s']:.2f}s  p95 {r['p95_s']:.2f}s  "
              f"{r['requests_per_s']:.3f} req/s  peak {r['peak_gib']:.2f} GiB", flush=True)

    return {
        "tag": f"student-latency-{adapter.rstrip('/').split('/')[-1]}",
        "adapter": adapter,
        "split": "test",
        "split_sha256": split_sha,
        "date": str(date.today()),
        "n": len(rows),
        "gpu": torch.cuda.get_device_name(0),
        "by_batch": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter repo id or local path")
    ap.add_argument("--latency", action="store_true", help="run the latency/throughput pass instead")
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16, help="batch for the quality pass (Phase 3 used 16)")
    ap.add_argument("--limit", type=int, help="cap rows, for a smoke test")
    args = ap.parse_args()

    import torch

    assert torch.cuda.is_available(), "no GPU visible — check the notebook's accelerator setting"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    split_sha = verify_frozen("test")
    rows = load_split("test", allow_test=True)
    assert len(rows) == 1000, f"test split is {len(rows)} rows, expected 1000"
    print(f"test split verified: {len(rows)} rows, sha256 {split_sha[:16]}… matches the manifest")

    from src.pii.eval import load_model

    model, tok = load_model(args.adapter)

    if args.latency:
        payload = latency(model, tok, latency_sample(rows, args.latency_n), args.adapter, split_sha)
    else:
        payload = quality(model, tok, rows[: args.limit] if args.limit else rows,
                          args.adapter, split_sha, args.batch_size)

    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{payload['tag']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote reports/raw/phase4/{path.name}", flush=True)


if __name__ == "__main__":
    main()

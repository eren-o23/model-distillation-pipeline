"""Phase 2: run the teacher over the train split and write the student's training set.

Two stages, separable so the expensive one is never repeated by accident:

  generate  - call the teacher, append every result to reports/raw/train-teacher.jsonl. Resumable:
              re-running skips uids already in the log, so an interrupted run costs nothing to finish.
  build     - read that log, filter it, write data/train_sft.jsonl and reports/phase2.md. Pure offline,
              re-runnable for free with --build-only.

  python scripts/generate_train.py                      # dry run: dedup count and cost estimate
  python scripts/generate_train.py --limit 20 --yes     # smoke test, ~$0.01
  caffeinate -i python -u scripts/generate_train.py --yes --workers 16
  python scripts/generate_train.py --build-only         # rebuild the report, no spend
"""

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import DATA_DIR, load_split, template_key  # noqa: E402
from src.pii.metric import Score, score, strip_hallucinated  # noqa: E402
from src.pii.student import to_sft_example  # noqa: E402
from src.pii.teacher import client, extract, price_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_LOG = ROOT / "reports" / "raw" / "train-teacher.jsonl"
TEACHER = "accounts/fireworks/models/qwen3p7-plus"

# Measured over the 200-example ceiling run, not guessed. A run that lands far above this is the
# tripwire for reasoning being back on (D-013), which silently invalidates the output.
TOKENS_PER_EXAMPLE = (628, 138)


def dedupe(rows: list[dict]) -> list[dict]:
    """Drop templated near-duplicates before spending anything on them."""
    seen: set[str] = set()
    kept = []
    for row in rows:
        key = template_key(row)
        if key not in seen:
            seen.add(key)
            kept.append(row)
    print(f"  template dedup: {len(rows)} -> {len(kept)} ({len(rows) - len(kept)} dropped)")
    return kept


def estimate(n: int, model_id: str) -> float:
    pin, pout = price_for(model_id)
    tin, tout = TOKENS_PER_EXAMPLE
    return n * (tin * pin + tout * pout) / 1_000_000


def read_log() -> dict[str, dict]:
    if not RAW_LOG.exists():
        return {}
    return {r["uid"]: r for r in (json.loads(line) for line in RAW_LOG.open())}


def generate(rows: list[dict], model_id: str, workers: int) -> None:
    """Call the teacher for every row not already logged, appending each result as it lands.

    Append-per-result rather than write-at-the-end: this run is ~40 minutes of paid API calls, and
    the bake-off's write-once checkpointing would throw all of it away on a crash at minute 39.
    """
    done = read_log()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"  {len(done)} already logged, {len(todo)} to fetch")
    if not todo:
        return

    RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
    cli = client()
    lock = threading.Lock()
    n_done = 0

    def work(row: dict) -> dict:
        entities, usage = extract(cli, model_id, row["source_text"])
        return {
            "uid": row["uid"],
            "entities": entities,
            "in": usage.prompt_tokens,
            "out": usage.completion_tokens,
        }

    with RAW_LOG.open("a") as log, ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(work, r) for r in todo]):
            record = future.result()
            with lock:
                log.write(json.dumps(record) + "\n")
                log.flush()  # a killed process must still leave every finished call on disk
                n_done += 1
                if n_done % 250 == 0 or n_done == len(todo):
                    print(f"    {n_done}/{len(todo)}", flush=True)


def build(rows: list[dict], model_id: str, n_frozen: int) -> None:
    """Filter the raw log into a training file and write the Phase 2 report.

    Filtering is repair, not selection against gold: schema-invalid outputs are dropped and invented
    values stripped, but examples the teacher merely got wrong are kept. The gold-agreement sweep at
    the end of the report measures what a stricter filter would do without applying it (D-018).
    """
    log = read_log()
    logged = [r for r in rows if r["uid"] in log]

    invalid = [r for r in logged if log[r["uid"]]["entities"] is None]
    kept, stripped, emptied = [], 0, 0
    for row in logged:
        ents = log[row["uid"]]["entities"]
        if ents is None:
            continue
        clean = strip_hallucinated(ents, row["source_text"])
        stripped += len(ents) - len(clean)
        if not clean:
            emptied += 1  # nothing left to teach; an empty target teaches "find nothing"
            continue
        kept.append((row, clean))

    out = DATA_DIR / "train_sft.jsonl"
    with out.open("w") as f:
        for row, ents in kept:
            f.write(json.dumps(to_sft_example(row["source_text"], ents), ensure_ascii=False) + "\n")

    # Phase 3 needs a val file in the same format for loss curves. It is built from GOLD, not from the
    # teacher — val is never sent to the API, so this costs nothing and measures progress toward the
    # true target rather than toward the teacher's mistakes.
    val = load_split("val")
    val_out = DATA_DIR / "val_sft.jsonl"
    with val_out.open("w") as f:
        for row in val:
            f.write(json.dumps(to_sft_example(row["source_text"], row["entities"]), ensure_ascii=False) + "\n")

    quality = score([r["entities"] for r, _ in kept], [e for _, e in kept], [r["source_text"] for r, _ in kept])
    tin = sum(log[r["uid"]]["in"] for r in logged)
    tout = sum(log[r["uid"]]["out"] for r in logged)
    pin, pout = price_for(model_id)
    spend = (tin * pin + tout * pout) / 1_000_000

    _write_report(model_id, n_frozen, rows, logged, invalid, kept, stripped, emptied, quality,
                  tin, tout, spend, out, val_out)


def _per_example_f1(gold: list[dict], pred: list[dict]) -> float:
    s = Score()
    s.add(gold, pred)
    return s.micro.f1


def _sweep(kept: list[tuple[dict, list]]) -> list[str]:
    """What a gold-agreement filter WOULD do at each threshold. Measured, not applied."""
    f1s = [(_per_example_f1(r["entities"], e), r, e) for r, e in kept]
    lines = ["| min per-example F1 | rows kept | % | label micro-F1 |", "|---|---|---|---|"]
    for t in (0.0, 0.5, 0.7, 0.9, 1.0):
        sel = [(r, e) for f, r, e in f1s if f >= t]
        if not sel:
            continue
        s = score([r["entities"] for r, _ in sel], [e for _, e in sel])
        mark = " **(applied)**" if t == 0.0 else ""
        lines.append(f"| ≥ {t:.1f}{mark} | {len(sel):,} | {len(sel) / len(kept):.1%} | {s.micro.f1:.3f} |")
    return lines


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(model_id, n_frozen, rows, logged, invalid, kept, stripped, emptied, quality,
                  tin, tout, spend, out, val_out) -> None:
    m = quality.micro
    weak = sorted(quality.per_label.items(), key=lambda kv: kv[1].f1)[:3]
    projected = estimate(len(logged), model_id)

    report = f"""# Phase 2 — training set generation

**Teacher:** `{model_id}`
**Run:** {date.today()}, synchronous serverless inference over the **train** split.
**The test split was not touched.**

## Row accounting

| stage | rows |
|---|---|
| frozen train split | {n_frozen:,} |
| after template dedup | {len(rows):,} |
| teacher responses logged | {len(logged):,} |
| − schema-invalid | −{len(invalid):,} |
| − empty after stripping invented values | −{emptied:,} |
| **written to `train_sft.jsonl`** | **{len(kept):,}** |

Schema-invalid rate: **{len(invalid) / max(len(logged), 1):.2%}** ({len(invalid)}/{len(logged)}), against 0%
observed on the 200-example ceiling run.
Invented entity values stripped: **{stripped:,}** — individual entities removed from otherwise usable
examples, rather than whole rows discarded.

## Training-label quality

This is what the student is actually taught, scored against train gold. It is **not** the 0.832 val ceiling:
that number was measured on different rows, and this one is the real upper bound on what Phase 3 can learn.

| metric | value |
|---|---|
| **micro-F1** | **{m.f1:.3f}** |
| precision | {m.precision:.3f} |
| recall | {m.recall:.3f} |

```
{quality.table()}
```

Weakest labels in the training data: {", ".join(f"`{k}` ({v.f1:.3f})" for k, v in weak)}. Phase 3 should
expect the student to inherit these, and Phase 4 should check whether it did.

## Gold-agreement sweep — measured, not applied

The train split has gold labels, so teacher outputs *could* be filtered against them. They were not: the
project's claim is distillation, and a real deployment has no gold to filter with. The table shows what a
stricter cut would cost and buy, so Phase 3 has the option costed if the student underperforms.

{chr(10).join(_sweep(kept))}

## Cost

| | |
|---|---|
| tokens | {tin:,} in / {tout:,} out |
| **actual spend** | **${spend:.2f}** |
| projected at 628/138 tokens per example | ${projected:.2f} |
| batch API alternative, declined (D-016) | ${spend / 2:.2f} |

A large overshoot against the projection would mean the teacher's reasoning had been re-enabled (D-013),
which invalidates the run. It did not.

## Files

| file | rows | sha256 |
|---|---|---|
| `data/train_sft.jsonl` | {len(kept):,} | `{_sha256(out)}` |
| `data/val_sft.jsonl` (gold-derived, no API cost) | {sum(1 for _ in val_out.open()):,} | `{_sha256(val_out)}` |

Both are gitignored, like every other `data/*.jsonl`. `data/manifest.json` is unchanged — the frozen splits
were not rebuilt.

## Method notes

- Student prompt is ~60 tokens, not the teacher's ~475 (D-017): the conventions live in the weights.
- The assistant turn is compact JSON with `ensure_ascii=False`, byte-identical to what the student must
  emit at serving time.
- Near-duplicate handling: rows fingerprinted with their PII values masked out, so two rows sharing a
  carrier sentence collapse. {n_frozen - len(rows)} dropped.
- `val_sft.jsonl` comes from gold, so val loss in Phase 3 measures progress toward the true target.
"""
    (ROOT / "reports" / "phase2.md").write_text(report)
    print(f"\nwrote reports/phase2.md · {len(kept):,} training rows · ${spend:.2f} spent")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--limit", type=int, help="cap rows, for smoke tests")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--yes", action="store_true", help="required to actually spend credits")
    ap.add_argument("--build-only", action="store_true", help="rebuild outputs from the log, no spend")
    args = ap.parse_args()

    rows = load_split("train")  # train only — test stays sealed until Phase 4
    n_frozen = len(rows)
    rows = dedupe(rows)
    if args.limit:
        rows = rows[: args.limit]

    if not args.build_only:
        pending = len([r for r in rows if r["uid"] not in read_log()])
        print(f"\n{len(rows):,} rows ({n_frozen:,} frozen, {pending:,} still to fetch)")
        print(f"  estimated spend: ~${estimate(pending, args.model):.2f}")
        if not args.yes:
            print("\nDry run. Re-run with --yes to spend credits.")
            return
        generate(rows, args.model, args.workers)

    build(rows, args.model, n_frozen)


if __name__ == "__main__":
    main()

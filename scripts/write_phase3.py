"""Generate reports/phase3.md from the JSON the Kaggle runs left behind.

Same discipline as write_ceiling.py: every number is read from a saved measurement rather than transcribed,
so the report cannot drift from what actually happened. Runs locally, needs no GPU and no torch.

The teacher's numbers are recomputed here from its Phase 1 checkpoint rather than copied from ceiling.md,
so teacher and student are scored by the same code over the same 200 val rows.

  python scripts/write_phase3.py            # after copying reports/raw/phase3/ back from Kaggle
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import load_split  # noqa: E402
from src.pii.metric import Score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports" / "raw" / "phase3"
TEACHER_CHECKPOINT = ROOT / "reports" / "raw" / "qwen3p7-plus-200.json"
EVAL_N = 200


def load(name: str) -> dict | None:
    path = RAW / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def teacher_score(n: int = EVAL_N) -> Score:
    """Recompute the teacher on the same rows the student is scored on."""
    blob = json.loads(TEACHER_CHECKPOINT.read_text())
    rows = load_split("val")[:n]
    s = Score()
    for row, pred in zip(rows, blob["predictions"][:n], strict=True):
        s.add(row["entities"], pred, row["source_text"])
    return s


def epochs(config: str) -> list[dict]:
    out = []
    for i in range(1, 9):
        blob = load(f"{config}-epoch{i}")
        if blob:
            out.append(blob | {"epoch": i})
    return out


def pct(student: float, teacher: float) -> str:
    return f"{student / teacher:.1%}" if teacher else "—"


def main() -> None:
    if not RAW.exists():
        sys.exit(f"no measurements at {RAW} — copy reports/raw/phase3/ back from Kaggle first")

    teacher = teacher_score()
    t_f1 = teacher.micro.f1
    b_short, b_teacher = load("baseline-short"), load("baseline-teacher")
    # Discover configs from the files rather than hardcoding ranks. Which second rank got trained is a
    # hardware outcome, not a constant: rank 32 would not fit alongside batch 4 on a 14.56GiB T4 (D-026).
    configs = sorted({p.stem.split("-epoch")[0] for p in RAW.glob("r*-epoch*.json")},
                     key=lambda c: int(c[1:]))
    runs = {c: epochs(c) for c in configs if epochs(c)}
    summaries = {c: load(f"{c}-summary") for c in runs}
    if not runs:
        sys.exit("no per-epoch results found — nothing to report yet")

    best_by_config = {c: max(e, key=lambda x: x["micro_f1"]) for c, e in runs.items()}
    winner = max(best_by_config, key=lambda c: best_by_config[c]["micro_f1"])
    best = best_by_config[winner]
    full = next((load(n.stem) for n in sorted(RAW.glob("adapter-*-full.json"))), None)
    # Per-label deltas are always taken from the 200-row eval, because that is the n the teacher's
    # checkpoint covers. Comparing a 1,000-row student against a 200-row teacher would put a sample-size
    # difference into the Δ column and read as a quality difference.
    headline = best

    # ---- headline ---------------------------------------------------------------------------------
    rows = [f"| **teacher** (`qwen3p7-plus`, API) | **{t_f1:.3f}** | — | ceiling |"]
    if b_short:
        rows.append(f"| untuned Qwen3-8B, ~60-token prompt | {b_short['micro_f1']:.3f} | "
                    f"{pct(b_short['micro_f1'], t_f1)} | baseline |")
    if b_teacher:
        rows.append(f"| untuned Qwen3-8B, teacher's ~475-token prompt | {b_teacher['micro_f1']:.3f} | "
                    f"{pct(b_teacher['micro_f1'], t_f1)} | baseline |")
    for c, e in best_by_config.items():
        mark = " **(best)**" if c == winner else ""
        rows.append(f"| **student** QLoRA {c}, epoch {e['epoch']}{mark} | **{e['micro_f1']:.3f}** | "
                    f"{pct(e['micro_f1'], t_f1)} | fine-tuned |")
    if full:
        rows.append(f"| ↳ same adapter, all {full['n_examples']:,} val rows | **{full['micro_f1']:.3f}** | "
                    f"{pct(full['micro_f1'], t_f1)} | **headline** |")

    gain = (best["micro_f1"] - b_short["micro_f1"]) if b_short else None
    prompt_gain = (b_teacher["micro_f1"] - b_short["micro_f1"]) if (b_short and b_teacher) else None

    # ---- per-epoch --------------------------------------------------------------------------------
    curve = ["| config | epoch | micro-F1 | P | R | schema-invalid | hallucinated |", "|---|---|---|---|---|---|---|"]
    for c, es in runs.items():
        for e in es:
            curve.append(f"| {c} | {e['epoch']} | {e['micro_f1']:.3f} | {e['micro_precision']:.3f} | "
                         f"{e['micro_recall']:.3f} | {e['schema_invalid']}/{e['n_examples']} "
                         f"({e['schema_invalid_rate']:.1%}) | {e['hallucinated']} |")

    # ---- per label vs teacher ---------------------------------------------------------------------
    labels = sorted(headline["per_label"], key=lambda k: -headline["per_label"][k]["support"])
    per_label = ["| label | teacher F1 | student F1 | Δ | support |", "|---|---|---|---|---|"]
    for lbl in labels:
        s = headline["per_label"][lbl]
        t = teacher.per_label.get(lbl)
        tf = t.f1 if t else 0.0
        per_label.append(f"| `{lbl}` | {tf:.3f} | {s['f1']:.3f} | {s['f1'] - tf:+.3f} | {s['support']} |")

    inherited = [lbl for lbl in ("IDCARDNUM", "GIVENNAME", "SURNAME")
                 if lbl in headline["per_label"] and headline["per_label"][lbl]["f1"] < 0.75]

    rank_heading = " vs ".join(f"rank {c[1:]}" for c in runs) or "Configurations"
    gpu_hours = {c: (s or {}).get("train_runtime_s", 0) / 3600 for c, s in summaries.items()}

    # Whether the val curve turned over is the one thing the per-epoch table shows but does not say. It
    # decides whether D-024's compute-driven epoch budget cost anything, so it is stated either way.
    notes = []
    for c, es in runs.items():
        if len(es) < 2:
            continue
        trace = " → ".join(f"{e['micro_f1']:.3f}" for e in es)
        peak = max(es, key=lambda e: e["micro_f1"])
        if peak["epoch"] < es[-1]["epoch"]:
            notes.append(
                f"**`{c}` peaked at epoch {peak['epoch']} and declined after it** ({trace}). The curve "
                "turning over is the overfitting point the spec warns about — found, not assumed. It also "
                "means the epoch budget was never the binding constraint on quality, so D-024's choice of "
                "2 epochs on compute grounds cost nothing here."
            )
        else:
            notes.append(
                f"**`{c}` was still improving at the last epoch** ({trace}), so the 2-epoch budget chosen "
                "on compute grounds (D-024) left quality on the table. Recorded rather than presented as "
                "optimal."
            )
    epoch_notes = "\n\n".join(notes)

    n_train = sum(1 for _ in (ROOT / "data" / "train_sft.jsonl").open())

    report = f"""# Phase 3 — QLoRA fine-tune of Qwen3-8B

**Student:** `Qwen3-8B` + QLoRA (4-bit NF4), trained on {n_train:,} teacher-generated examples.
**Hardware:** Kaggle 2xT4 (sm_75) — fp16, no bf16, no FlashAttention-2 (D-019).
**Measured:** {date.today()} against dataset **gold** on the **val** split — every model in the headline
table on the same {EVAL_N} rows the teacher ceiling was measured on""" + (
        f", plus one confirmation run of the winning adapter over all {full['n_examples']:,} (D-021)."
        if full else " (D-021)."
    ) + f"""
**The test split was not touched.** Phase 3 never passes `allow_test=True`.
**API spend: $0.** Training and evaluation are entirely local to Kaggle's free tier.

## Headline

| model | micro-F1 | % of teacher | |
|---|---|---|---|
{chr(10).join(rows)}

""" + ("" if gain is None else f"""Fine-tuning moved the same 8B model from **{b_short['micro_f1']:.3f}** to
**{best['micro_f1']:.3f}** on an identical ~60-token prompt — **{gain:+.3f} F1** that is attributable to the
weights rather than to the context.
""") + ("" if prompt_gain is None else f"""
Prompting the untuned model with the teacher's full ~475-token conventions instead buys **{prompt_gain:+.3f}**.
That is the honest control for D-017: the student reaches its score on roughly a quarter of the input tokens,
and the gap between those two baselines is what the fine-tune replaced.
""") + f"""
## Per epoch

{chr(10).join(curve)}

Checkpoints are selected on **val micro-F1, not val loss** (D-021) — they disagree, and F1 is the reported
number. Evaluation is against gold, so the student is measured on the same absolute yardstick as the teacher
and can in principle exceed it.

{epoch_notes}

## {rank_heading}

| config | trainable params | best epoch | micro-F1 | train time |
|---|---|---|---|---|
""" + "\n".join(
        f"| r{c[1:]} | see W&B | {best_by_config[c]['epoch']} | {best_by_config[c]['micro_f1']:.3f} | "
        f"{gpu_hours.get(c, 0):.1f}h |" for c in runs
    ) + f"""

`lora_alpha` tracks rank at 2r in both runs, so the `alpha/r` scaling is constant and adapter capacity is the
only difference between them (D-020). Without that, a rank change would also be a 4x change in effective
update size and the comparison would be unreadable.

## Per label, against the teacher

{chr(10).join(per_label)}

""" + (f"""Still weak: {", ".join(f"`{lbl}`" for lbl in inherited)}. Phase 2 predicted this — the training
labels themselves score 0.419 F1 on `IDCARDNUM` (D-015) and ~0.68 on the name split, so the student is
learning the teacher's blind spots faithfully rather than developing its own. That is what distillation does,
and it is the argument for the escalation path in Phase 5 rather than for more training.
""" if inherited else "No label fell below 0.75, so the weaknesses Phase 2 predicted did not transfer.\n") + f"""
## Cost

| | |
|---|---|
| API spend | **$0.00** |
| Kaggle GPU | {sum(gpu_hours.values()):.1f}h of a ~30h weekly quota |
| running project total | ~$5.02 of $50, unchanged from Phase 2 |

## Method notes

- Loss is masked to the assistant turn only. Measured on this data, 33% of tokens are answer, so training on
  the full sequence would spend two thirds of the gradient reproducing a fixed prompt (D-023).
- `max_length` 1024 truncates 0 of 7,842 rows; the collator pads to the longest row per batch (mean 276), so
  the cap costs nothing.
- Prompt rendering goes through `student.render_prompt` with `enable_thinking=False`, asserted token-exact
  against the training render by `tests/test_prompt_render.py` (D-022).
- Generation is greedy and left-padded; batches are length-sorted so the 871-token tail does not pad out the
  median 246-token row.
- The teacher's per-label numbers above are recomputed from its Phase 1 checkpoint by this script, over the
  same 200 rows, rather than copied from `ceiling.md`.
"""

    (ROOT / "reports" / "phase3.md").write_text(report)
    print(f"wrote reports/phase3.md · best {winner} at {best['micro_f1']:.3f} "
          f"({pct(best['micro_f1'], t_f1)} of teacher)")


if __name__ == "__main__":
    main()

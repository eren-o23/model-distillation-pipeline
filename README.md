# Model Distillation Pipeline

Distilling a PII-detection capability from a large open-weight teacher into a 7B student, then benchmarking
the two on quality, cost, and latency to find the **break-even request volume** — the traffic level above
which self-hosting the student beats paying per-token for the teacher.

**Status:** Phase 1 of 5 — task locked, teacher ceiling in progress.

---

## The question this project answers

Not "can a small model do this task" (it can), but **at what request volume does owning the model beat
renting it** — including idle GPU time, and including the cost of escalating the cases the student gets wrong.

## Approach

| | |
|---|---|
| **Task** | PII entity detection over English text — 12 entity classes |
| **Dataset** | [`ai4privacy/pii-masking-openpii-1.5m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) (CC-BY-4.0) |
| **Teacher** | Large open-weight Qwen model, served on Fireworks |
| **Student** | `Qwen2.5-7B-Instruct` + QLoRA |
| **Metric** | Micro and per-label precision / recall / F1 on `(label, value)` pairs |

Teacher and student are from the same model family on purpose: the comparison then isolates the effect of
*scale*, rather than confounding it with tokenizer and pretraining differences.

## Results

_Phase 1 in progress. Teacher ceiling, three-axis benchmark, and break-even volume land here._

## Cost

Every API call in this project is metered and reported — cost transparency is part of what is being
demonstrated, and the break-even number is meaningless without it.

| Phase | Spend |
|---|---|
| Phase 1 — teacher bake-off | _pending_ |
| **Total** | _pending_ |

Budget ceiling: **$50** in Fireworks credits. Fine-tuning runs on Kaggle's free tier (2×T4).

## Evaluation discipline

- **Splits are frozen before any tuning**, written with a SHA256 manifest, and drawn from disjoint source
  splits so `train` / `val` / `test` cannot overlap.
- **The test set is not touched until Phase 4.** `load_split("test")` raises unless explicitly overridden.
- **Prompt development happens on `val` only** — tuning prompts against the test set would contaminate the
  headline number.
- **Training labels come from the teacher; all evaluation is against dataset gold.** Teacher and student are
  therefore scored on the same absolute yardstick, which also means the student can in principle exceed the
  teacher rather than being capped by it.

## Repo layout

```
docs/decisions.md      decision log — what was chosen, over what, and why
src/pii/               data prep, metric, teacher client
scripts/               split construction, teacher bake-off
reports/               bake-off results, ceiling report
tests/                 metric tests
```

## Setup

```bash
pip install -r requirements.txt
echo "FIREWORKS_API_KEY=..." > .env   # gitignored
python scripts/build_splits.py
```

## Design decisions

The reasoning behind every significant choice — including the ones that went against the obvious option — is
logged in [docs/decisions.md](docs/decisions.md).

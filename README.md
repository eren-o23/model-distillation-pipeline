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

**Phase 1 complete.** Teacher ceiling established; student benchmark and break-even volume land here next.

| | |
|---|---|
| **Teacher** | `qwen3p7-plus` (Fireworks) |
| **Teacher ceiling** | **0.832 micro-F1** on held-out val ([reports/ceiling.md](reports/ceiling.md)) |
| schema-invalid | 0.0% |
| hallucinated values | 0 |

Two candidate teachers were measured before committing budget
([reports/teacher-comparison.md](reports/teacher-comparison.md)): `deepseek-v4-pro` scored +0.005 F1 for 3.4x
the cost, so `qwen3p7-plus` won on economics.

Strongest labels are `EMAIL` (1.000), `DATE` (0.983) and `TELEPHONENUM` (0.944). The ceiling is held down by
`GIVENNAME`/`SURNAME` (~0.68), where the dataset's own given/family-name boundary is genuinely ambiguous on
multicultural names, and by `IDCARDNUM` (0.377) — see the decision log for why that one resisted fixing.

## Cost

Every API call in this project is metered and reported — cost transparency is part of what is being
demonstrated, and the break-even number is meaningless without it.

| Phase | Spend |
|---|---|
| Phase 1 — teacher bake-off and ceiling | ~$1.50 |
| — of which wasted on an invalidated run | ~$1.00 |
| **Total to date** | **~$1.50** |
| Phase 2 projection — 8,000 examples (batch) | $1.88 |

Roughly $1 of that was spent measuring truncation instead of capability, before the reasoning-token bug was
found. It is reported rather than quietly dropped, because a cost analysis that only counts the runs that
worked is not a cost analysis.

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

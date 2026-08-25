# Phase 2 — training set generation

**Teacher:** `accounts/fireworks/models/qwen3p7-plus`
**Run:** 2026-08-25, synchronous serverless inference over the **train** split.
**The test split was not touched.**

## Row accounting

| stage | rows |
|---|---|
| frozen train split | 8,000 |
| after template dedup | 7,992 |
| teacher responses logged | 7,992 |
| − unrecovered API failures (429/timeout, nothing billed) | −0 |
| − schema-invalid | −0 |
| − empty after stripping invented values | −150 |
| **written to `train_sft.jsonl`** | **7,842** |

Schema-invalid rate: **0.00%** (0 of
7,992 responses that actually reached the model), against 0% on the 200-example
ceiling run. Invented entity values stripped: **18** — individual entities removed from otherwise
usable examples, rather than whole rows discarded. The 150 dropped rows are ones where the teacher
returned nothing at all; an empty target would teach the student to find nothing.

**API failures are counted separately on purpose.** A 429 or a timeout is a transport failure that never
reached the model and was never billed, so folding it into the schema-invalid rate would report a teacher
quality problem that did not occur. The first pass at 16 workers hit **684** of these — 8.6% of the run,
which would have been published as a teacher defect. Re-fetching them at 8 workers recovered all 684 with
zero errors, confirming the cause was local concurrency rather than anything about the model.

## Training-label quality

This is what the student is actually taught, scored against train gold. It is **not** the 0.832 val ceiling:
that number was measured on different rows, and this one is the real upper bound on what Phase 3 can learn.

| metric | value |
|---|---|
| **micro-F1** | **0.832** |
| precision | 0.849 |
| recall | 0.816 |

```
label                      P       R      F1      n
GIVENNAME              0.703   0.701   0.702   7503
DATE                   0.989   0.984   0.986   7353
SURNAME                0.685   0.712   0.698   6353
EMAIL                  0.990   0.997   0.993   4556
CITY                   0.784   0.799   0.791   4175
TELEPHONENUM           0.906   0.990   0.946   3306
STREET                 0.880   0.934   0.906   2940
ZIPCODE                0.920   0.909   0.915   2556
IDCARDNUM              0.753   0.291   0.419   1882
CREDITCARDNUMBER       0.977   0.776   0.865   1850
TAXNUM                 0.983   0.675   0.801   1475
SOCIALNUM              0.923   0.550   0.689   1354
--------------------------------------------------
MICRO                  0.849   0.816   0.832  45303

examples: 7842  schema-invalid: 0 (0.0%)  hallucinated values: 0
```

Weakest labels in the training data: `IDCARDNUM` (0.419), `SOCIALNUM` (0.689), `SURNAME` (0.698). Phase 3 should
expect the student to inherit these, and Phase 4 should check whether it did.

## Gold-agreement sweep — measured, not applied

The train split has gold labels, so teacher outputs *could* be filtered against them. They were not: the
project's claim is distillation, and a real deployment has no gold to filter with. The table shows what a
stricter cut would cost and buy, so Phase 3 has the option costed if the student underperforms.

| min per-example F1 | rows kept | % | label micro-F1 |
|---|---|---|---|
| ≥ 0.0 **(applied)** | 7,842 | 100.0% | 0.832 |
| ≥ 0.5 | 7,315 | 93.3% | 0.857 |
| ≥ 0.7 | 5,938 | 75.7% | 0.909 |
| ≥ 0.9 | 3,909 | 49.8% | 0.982 |
| ≥ 1.0 | 3,295 | 42.0% | 1.000 |

## Cost

| | |
|---|---|
| tokens | 4,901,418 in / 974,137 out |
| **actual spend** | **$3.52** |
| projected at 628/138 tokens per example | $3.77 |
| batch API alternative, declined (D-016) | $1.76 |

A large overshoot against the projection would mean the teacher's reasoning had been re-enabled (D-013),
which invalidates the run. It did not.

## Files

| file | rows | sha256 |
|---|---|---|
| `data/train_sft.jsonl` | 7,842 | `6c51de7a559c6343b1a3bb66872a86fc6c2539f141005587fc3cb8220eae24e3` |
| `data/val_sft.jsonl` (gold-derived, no API cost) | 1,000 | `128122193796ff86ac40e12c4ca16884c4f3aa9a663ad2648ac7306a5c49003d` |

Both are gitignored, like every other `data/*.jsonl`. `data/manifest.json` is unchanged — the frozen splits
were not rebuilt.

## Method notes

- Student prompt is ~60 tokens, not the teacher's ~475 (D-017): the conventions live in the weights.
- The assistant turn is compact JSON with `ensure_ascii=False`, byte-identical to what the student must
  emit at serving time.
- Near-duplicate handling: rows fingerprinted with their PII values masked out, so two rows sharing a
  carrier sentence collapse. 8 dropped.
- `val_sft.jsonl` comes from gold, so val loss in Phase 3 measures progress toward the true target.

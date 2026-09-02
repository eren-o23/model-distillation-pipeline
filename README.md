# Model Distillation Pipeline

Distilling a PII-detection capability from a large open-weight teacher into an 8B student, then benchmarking
the two on quality, cost, and latency to find the **break-even request volume** — the traffic level above
which self-hosting the student beats paying per-token for the teacher.

**Status:** Phase 3 of 5 complete — the student matches the teacher at 0.833 micro-F1. Benchmarking next.

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
| **Student** | `Qwen3-8B` + QLoRA (4-bit NF4), LoRA rank 8 |
| **Metric** | Micro and per-label precision / recall / F1 on `(label, value)` pairs |

Teacher and student share a lineage but **not a generation** — Qwen3.7 was never released as open weights,
so a true same-generation pair is impossible. The "isolates scale" claim was therefore dropped rather than
quietly kept (D-011).

## Results

**Phases 1–3 complete.** The 8B student **matches the teacher** on held-out validation, at roughly a
quarter of the input tokens. The sealed test set and the break-even volume land here next.

### The student ([reports/phase3.md](reports/phase3.md))

| model | micro-F1 | schema-invalid |
|---|---|---|
| **teacher** `qwen3p7-plus` (API) | 0.832 | 0.0% |
| untuned Qwen3-8B, ~60-token prompt | 0.656 | 0.5% |
| untuned Qwen3-8B, teacher's ~475-token prompt | 0.725 | 0.5% |
| **student** Qwen3-8B + QLoRA rank 8 | **0.833** | **0.0%** |

The student is scored on all 1,000 validation rows, the teacher on 200 (D-021), so the two are not
measured at equal precision — **matches** is the defensible claim, not *beats*. Phase 4 settles it on the
sealed test set, where re-measuring the teacher at the same n costs about $0.24.

Two baselines were measured before training, and they are what make the result readable. Fine-tuning is worth
**+0.173** over the same model on the same short prompt. Prompting alone — handing the untuned model the
teacher's full 475-token conventions — buys only +0.069, and costs 4x the input tokens every request. That
gap is the evidence behind D-017, and it feeds the break-even number directly.

Schema-invalid output fell from 0.5% to **0 in 1,000**. That is a reliability gain on top of accuracy, and it
sets the floor for Phase 5's escalation rate.

**Rank 8 against rank 16** is a null result, reported as one: 0.829 vs 0.823 on 200 rows, a 0.005 gap against
a ~0.011 standard error. Twice the adapter capacity bought nothing measurable. Rank 32 was the planned second
configuration but does not fit alongside batch 4 on a 16GB T4 (D-026).

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

The student inherited that last one precisely: **`IDCARDNUM` at 0.901 precision but 0.285 recall** across
1,000 rows. It learned to be conservative because the teacher taught it to be (D-015), which is the clearest
concrete argument in the project for Phase 5's escalation path.

### Training set ([reports/phase2.md](reports/phase2.md))

| | |
|---|---|
| **Training examples** | **7,842** |
| Label quality vs. train gold | **0.832 micro-F1** |
| schema-invalid | 0.00% |
| invented values stripped | 18 entities |

The training labels scoring 0.832 against train gold — matching the 0.832 val ceiling almost exactly — is a
useful consistency check: the teacher performs the same on both splits, so the reservoir-sampled splits are
drawing from one distribution rather than two.

Teacher outputs are **repaired, not filtered against gold**: unparseable outputs and invented values are
dropped, but examples the teacher merely got wrong are kept. Filtering by gold agreement would train the
student only on what the teacher found easy, and a real deployment generating data from a frontier model has
no gold to filter with. The report costs that option out anyway — at a 0.7 agreement threshold, 76% of rows
survive at 0.909 label quality — so Phase 3 can take it if the student underperforms.

The student trains on a **~60-token prompt** rather than the teacher's ~475-token one: the annotation
conventions live in the weights, not the context. That cuts input tokens per request roughly 4x, which feeds
the break-even number directly (D-017).

## Cost

Every API call in this project is metered and reported — cost transparency is part of what is being
demonstrated, and the break-even number is meaningless without it.

| Phase | Spend |
|---|---|
| Phase 1 — teacher bake-off and ceiling | ~$1.50 |
| — of which wasted on an invalidated run | ~$1.00 |
| Phase 2 — 7,992 examples generated | **$3.52** |
| Phase 3 — fine-tuning and evaluation | **$0.00** |
| — projected beforehand | $3.77 |
| — batch API alternative, declined | $1.76 |
| **Total to date** | **~$5.02** |

Phase 2 ran synchronously rather than through the batch API, paying $1.76 more to keep an expensive run
interactive and resumable (D-016). The declined saving is reported rather than omitted — a cost analysis that
only shows the cheapest path you *could* have taken is not a cost analysis either.

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
src/pii/               data prep, metric, teacher client, student prompt format
scripts/               split construction, teacher bake-off, training-set generation
reports/               bake-off results, ceiling report, Phase 2 report + raw per-row logs
tests/                 metric and training-set tests
```

## Setup

```bash
pip install -r requirements.txt
echo "FIREWORKS_API_KEY=..." > .env   # gitignored
python scripts/build_splits.py                    # ~10 min, one full pass over 1.6M rows
python scripts/generate_train.py                  # dry run: prints the cost estimate, spends nothing
python scripts/generate_train.py --yes            # the real generation run
```

## Design decisions

The reasoning behind every significant choice — including the ones that went against the obvious option — is
logged in [docs/decisions.md](docs/decisions.md).

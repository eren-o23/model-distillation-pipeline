# Phase 4 — Student against teacher on quality, cost and latency

**Measured:** 2026-09-03 on the **sealed test split** — 1,000 rows, sha256 `908f52f75ed401f1…`,
verified against the manifest Phase 1 froze it with.
**Opened once.** Both models were scored on these same rows by the same metric, and nothing was tuned
against them: every prompt, batch size and checkpoint decision was made on `val` in Phases 1-3.
**Student:** `erenrosman/pii-qwen3-8b-lora-r8` on Kaggle's T4 — selected on **val** in Phase 3 (rank 8 at 0.829), not on the rows below.
**Teacher:** `qwen3p7-plus` on Fireworks.
**Phase 4 API spend: $0.46.**

## The three axes

| | quality (micro-F1) | cost / 1k requests | latency p50 | latency p95 |
|---|---|---|---|---|
| **teacher** | 0.822 | $0.44 | 0.73s | 1.19s |
| **student** | 0.823 | $0.43 at 100% utilisation | 27.11s | 48.19s |

Quality is measured at equal n on identical rows. Cost and latency are **not** like-for-like in the same
way and the sections below say exactly how: the teacher is a hosted API billed per token, the student is
one rented T4 running HF `generate` with static batching, which is the pre-vLLM floor rather than a
serving number (D-030).

**On this serving stack there is no break-even volume.** The student costs $0.43 per 1,000 requests on a fully saturated T4 against the teacher's $0.44 — 100% of the API's price at 100% utilisation, and *more* than the API at any realistic one ($0.87 at 50%). Self-hosting is not cheaper here at one request per day or at ten million, and it is 37x slower per request.

That is a real result rather than a setback, and it locates the problem precisely: the student's quality is not the constraint — it matches the teacher — and neither is the hardware price. **The constraint is throughput.** At 0.336 req/s, HF `generate` leaves the card idle between batches and pads every request to the longest in its batch. Phase 5's vLLM benchmark needs **0.67 req/s** for the student to merely match API pricing at 50% utilisation, and **6.7 req/s** for the tenfold advantage this project set out to find. Those are the numbers to beat, and they come from measurement rather than from hope.

## Axis 1 — Quality

| model | micro-F1 | precision | recall | schema-invalid | hallucinated |
|---|---|---|---|---|---|
| teacher | 0.822 | 0.845 | 0.800 | 0/1000 (0.0%) | 2 |
| student | 0.823 | 0.842 | 0.805 | 0/1000 (0.0%) | 3 |

**The two are statistically indistinguishable.** The gap is +0.001 micro-F1 and the 95% confidence interval on it, [-0.006, +0.007], spans zero — so **"matches the teacher" is the claim the evidence supports, and "beats" is not**, even now that both are measured on the same 1,000 rows. The student was ahead in 60% of 2,000 resamples.

The interval comes from a paired bootstrap: 2,000 resamples of the 1,000 rows, both models
re-scored on each draw. Pairing is what makes it tight — the same rows are hard for both models, and
resampling them together cancels that shared difficulty instead of counting it twice.

### Per label

| label | teacher F1 | student F1 | Δ | student P | student R | support |
|---|---|---|---|---|---|---|
| `DATE` | 0.987 | 0.994 | +0.006 | 0.993 | 0.995 | 935 |
| `GIVENNAME` | 0.674 | 0.667 | -0.007 | 0.670 | 0.665 | 892 |
| `SURNAME` | 0.675 | 0.666 | -0.009 | 0.660 | 0.673 | 792 |
| `EMAIL` | 0.988 | 0.986 | -0.002 | 0.983 | 0.989 | 568 |
| `CITY` | 0.786 | 0.787 | +0.002 | 0.783 | 0.792 | 538 |
| `TELEPHONENUM` | 0.947 | 0.921 | -0.026 | 0.855 | 0.998 | 407 |
| `STREET` | 0.915 | 0.912 | -0.003 | 0.892 | 0.933 | 373 |
| `ZIPCODE` | 0.914 | 0.928 | +0.014 | 0.915 | 0.942 | 310 |
| `CREDITCARDNUMBER` | 0.862 | 0.910 | +0.047 | 0.979 | 0.849 | 279 |
| `IDCARDNUM` | 0.422 | 0.415 | -0.006 | 0.802 | 0.280 | 275 |
| `TAXNUM` | 0.767 | 0.765 | -0.003 | 0.977 | 0.628 | 207 |
| `SOCIALNUM` | 0.672 | 0.683 | +0.011 | 1.000 | 0.518 | 164 |

**`IDCARDNUM` transferred exactly as D-015 predicted**: 0.802 precision against
0.280 recall. The student learned to be conservative because the teacher taught it to be —
the training labels themselves score 0.419 F1 on this class. It is the clearest concrete argument for
Phase 5's escalation path: a router that sends low-confidence `IDCARDNUM` cases back to the teacher buys
recall the student cannot be trained into without reopening D-006 and rebuilding the splits.

### val → test

| model | val | test | Δ |
|---|---|---|---|
| student (rank 8) | 0.833 (1,000 val) | 0.823 (1,000 test) | -0.010 |
| teacher | 0.832 (200 val) | 0.822 (1,000 test) | -0.010 |

Prompt development happened on `val` across three phases, so a large drop here would be the signature of that leaking into the headline. It did not: both models move by less than 0.03 F1, which is what two draws from one distribution look like.

## rank 8 vs rank 16, on the sealed test set

| config | micro-F1 | P | R | schema-invalid |
|---|---|---|---|---|
| rank 16 | 0.826 | 0.844 | 0.809 | 0/1000 |
| rank 8 | 0.823 | 0.842 | 0.805 | 0/1000 |

The gap is 0.003 against a standard error of about 0.005 on 5,740 scored entities. **D-027's null result holds at 1,000 rows** — five times the sample that produced it, on data neither configuration has ever seen. Two ranks differing by 2x in capacity are not separable on this task, and rank 8 stays selected on cost — a decision made on val, which is why the ordering above does not change it.

## Axis 2 — Cost per 1,000 requests

| system | basis | 100% utilisation | 50% | 25% |
|---|---|---|---|---|
| **teacher** (`qwen3p7-plus` API) | per-token, no idle cost | **$0.44** | $0.44 | $0.44 |
| **student** (T4, batch 16) | 0.336 req/s measured | **$0.43** | $0.87 | $1.74 |

**Teacher:** measured, not estimated — $0.4366 for 1,000 requests at
614 input and 119
output tokens each. An API has no idle cost, so its three columns are identical; that is precisely the
property the student has to beat on volume.

**Student:** priced at AWS `g4dn.xlarge` (1x T4 16GB), us-east-1 on-demand, [$0.526/hr](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), checked 2026-09-02. Kaggle's GPU is free, which would make the student cost $0 and
win at any volume — a number that flatters self-hosting by hiding the hardware, which is a named failure
mode of this comparison. The utilisation columns charge for idle time: a GPU billed by the hour costs the
same whether or not requests arrive, so the 25% column is the honest one for bursty traffic.

**This student number is an upper bound.** HF `generate` with static batching leaves the card idle
between batches and pads every request to the longest in its batch. vLLM's continuous batching and paged
attention is what Phase 5 measures, and it will lower this figure — so the break-even volume computed
from this table would be pessimistic, and is deliberately not computed here.

## Axis 3 — Latency

| system | concurrency | p50 | p95 |
|---|---|---|---|
| teacher (API) | 1 | 0.56s | 0.85s |
| teacher (API) | 8 | 0.73s | 1.19s |
| student (T4, batch 1) | 1 | 13.32s | 30.41s |
| student (T4, batch 8) | 8 | 27.11s | 48.19s |

Concurrency means different things on the two sides and the mapping is stated rather than implied: for
the teacher it is in-flight HTTP requests; for the student it is batch size, because static batching
returns every request in a batch when its slowest member finishes. Both were timed on the **same seeded
100 test rows**, in arrival order rather than length-sorted — sorting is right for a
bulk eval and wrong for a latency measurement, since it reports a workload where every request happens to
arrive beside others of its own length.

### Throughput curve

| batch | p50 | p95 | req/s | peak VRAM | $/1k @ 100% |
|---|---|---|---|---|---|
| 1 | 13.32s | 30.41s | 0.067 | 5.94 GiB | $2.18 |
| 8 | 27.11s | 48.19s | 0.261 | 6.62 GiB | $0.56 |
| 16 | 38.61s | 63.29s | 0.336 | 7.50 GiB | $0.43 |
| 32 | 96.30s | 100.63s | 0.336 | 9.26 GiB | $0.44 |

This curve is what Axis 2 is amortised over, and what Phase 5 re-measures under vLLM.

## Method notes

- Both models are scored against dataset **gold** by `src/pii/metric.py`, never against each other.
- The student's numbers come from `src.pii.eval.evaluate()` unchanged — the same function that produced
  every Phase 3 number, so the val → test difference above is a data difference, not a harness one.
- `reasoning_effort: "none"` remains set on the teacher (D-013). The run's actual spend is checked
  against its projection at run time; a large overshoot means reasoning is back on and the numbers are
  measuring truncation.
- The test split was verified by sha256 against `data/manifest.json` before either model saw it, and
  every saved measurement records the hash it was taken against.
- Every number here is generated from `reports/raw/phase4/*.json` by this script.

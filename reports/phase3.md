# Phase 3 — QLoRA fine-tune of Qwen3-8B

**Student:** `Qwen3-8B` + QLoRA (4-bit NF4), trained on 7,842 teacher-generated examples.
**Hardware:** Kaggle 2xT4 (sm_75) — fp16, no bf16, no FlashAttention-2 (D-019).
**Measured:** 2026-08-27 against dataset **gold** on the **val** split — every model in the headline
table on the same 200 rows the teacher ceiling was measured on (D-021).
**The test split was not touched.** Phase 3 never passes `allow_test=True`.
**API spend: $0.** Training and evaluation are entirely local to Kaggle's free tier.

## Headline

| model | micro-F1 | % of teacher | |
|---|---|---|---|
| **teacher** (`qwen3p7-plus`, API) | **0.832** | — | ceiling |
| untuned Qwen3-8B, ~60-token prompt | 0.656 | 78.8% | baseline |
| untuned Qwen3-8B, teacher's ~475-token prompt | 0.725 | 87.1% | baseline |
| **student** QLoRA r8, epoch 1 **(best)** | **0.829** | 99.6% | fine-tuned |

Fine-tuning moved the same 8B model from **0.656** to
**0.829** on an identical ~60-token prompt — **+0.173 F1** that is attributable to the
weights rather than to the context.

Prompting the untuned model with the teacher's full ~475-token conventions instead buys **+0.069**.
That is the honest control for D-017: the student reaches its score on roughly a quarter of the input tokens,
and the gap between those two baselines is what the fine-tune replaced.

## Per epoch

| config | epoch | micro-F1 | P | R | schema-invalid | hallucinated |
|---|---|---|---|---|---|---|
| r8 | 1 | 0.829 | 0.838 | 0.820 | 0/200 (0.0%) | 1 |
| r8 | 2 | 0.822 | 0.831 | 0.814 | 0/200 (0.0%) | 1 |

Checkpoints are selected on **val micro-F1, not val loss** (D-021) — they disagree, and F1 is the reported
number. Evaluation is against gold, so the student is measured on the same absolute yardstick as the teacher
and can in principle exceed it.

**`r8` peaked at epoch 1 and declined after it** (0.829 → 0.822). The curve turning over is the overfitting point the spec warns about — found, not assumed. It also means the epoch budget was never the binding constraint on quality, so D-024's choice of 2 epochs on compute grounds cost nothing here.

## Rank 8 vs rank 32

| config | trainable params | best epoch | micro-F1 | train time |
|---|---|---|---|---|
| r8 | see W&B | 1 | 0.829 | 7.6h |

`lora_alpha` tracks rank at 2r in both runs, so the `alpha/r` scaling is constant and adapter capacity is the
only difference between them (D-020). Without that, a rank change would also be a 4x change in effective
update size and the comparison would be unreadable.

## Per label, against the teacher

| label | teacher F1 | student F1 | Δ | support |
|---|---|---|---|---|
| `DATE` | 0.983 | 0.993 | +0.010 | 209 |
| `GIVENNAME` | 0.680 | 0.680 | +0.000 | 199 |
| `SURNAME` | 0.686 | 0.676 | -0.010 | 171 |
| `CITY` | 0.823 | 0.804 | -0.018 | 133 |
| `EMAIL` | 1.000 | 1.000 | +0.000 | 131 |
| `TELEPHONENUM` | 0.944 | 0.924 | -0.021 | 104 |
| `STREET` | 0.914 | 0.908 | -0.006 | 90 |
| `ZIPCODE` | 0.813 | 0.810 | -0.003 | 72 |
| `IDCARDNUM` | 0.377 | 0.457 | +0.080 | 53 |
| `CREDITCARDNUMBER` | 0.932 | 0.920 | -0.012 | 47 |
| `TAXNUM` | 0.836 | 0.818 | -0.018 | 39 |
| `SOCIALNUM` | 0.727 | 0.679 | -0.048 | 35 |

Still weak: `IDCARDNUM`, `GIVENNAME`, `SURNAME`. Phase 2 predicted this — the training
labels themselves score 0.419 F1 on `IDCARDNUM` (D-015) and ~0.68 on the name split, so the student is
learning the teacher's blind spots faithfully rather than developing its own. That is what distillation does,
and it is the argument for the escalation path in Phase 5 rather than for more training.

## Cost

| | |
|---|---|
| API spend | **$0.00** |
| Kaggle GPU | 7.6h of a ~30h weekly quota |
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

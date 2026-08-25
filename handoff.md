# Handoff — model-distillation-pipeline

_Last updated: 2026-08-25_

---

## Goal

Portfolio project for AI/ML Engineering grad applications. Distil a PII-detection capability from a large
API-served teacher into a locally fine-tuned 8B student, then benchmark both on quality, cost and latency to
produce the deliverable that actually differentiates the project: **a break-even request volume** — the traffic
level above which self-hosting the student beats paying per-token for the teacher — plus a served endpoint that
escalates back to the teacher when the student is unsure. Five phases; Phase 1 is complete.

---

## Current State

**Phase 1 complete and verified.**

- `src/pii/metric.py` — multiset-per-label scoring of `(label, value)` pairs. Verified: `pytest
  tests/test_metric.py` → 9/9 pass.
- `src/pii/data.py` — label scope, reservoir-sampled splits, sealed test set. Verified by running
  `scripts/build_splits.py`: 8,000/1,000/1,000 written, uid-disjointness asserted across all three pairs,
  `load_split("test")` confirmed to raise without `allow_test=True`.
- `src/pii/teacher.py` — Fireworks client. Verified by live calls against the API.
- **Teacher ceiling: 0.832 micro-F1** (`qwen3p7-plus`, 200 val examples, 0% schema-invalid, 0 hallucinated
  values). In `reports/ceiling.md`, generated from the saved checkpoint rather than transcribed.
- Spend to date ~$1.50 of a $50 Fireworks budget.

Not yet written: any Phase 2+ code. There is **no batch-API implementation** — see Open Questions.

---

## Key Invariants

- **Python 3.12 required.** The system `python3` on this machine is 3.9.6 and **cannot even import**
  `src/pii/metric.py` (`X | None` in a signature raises `TypeError` at def time). Use `.venv/bin/python`,
  built from `/opt/anaconda3/bin/python3.12`. A bare `python3` will mislead you.
- **`reasoning_effort: "none"` on teacher calls is load-bearing.** `qwen3p7-plus` is a reasoning model; left
  on it emits ~2,459 output tokens (~1,900 of them thinking), overruns the token cap, truncates its own JSON,
  and the score drops from 0.832 to 0.548 with 33.5% unparseable. Removing this line silently invalidates
  every measurement.
- **The `SYSTEM` prompt encodes dataset annotation conventions**, not stylistic preferences. The rules about
  grouping given names into one span, excluding building numbers from `STREET`, copying dates verbatim
  including the time component, and restricting the label list are each worth multiple F1 points. Editing
  them casually will move the ceiling.
- **`run_bakeoff.py` silently reuses checkpoints.** If `reports/raw/<model>-<n>.json` exists it is loaded
  instead of calling the API. To force a genuine re-run you must move or delete that file — otherwise you
  will "measure" a stale prompt and not notice.
- **`data/*.jsonl` are gitignored and not in the repo.** Only `data/manifest.json` is tracked. Rebuilding
  requires `scripts/build_splits.py`, which does a **full pass over ~1.6M rows (~10 min)** and reproduces the
  same splits only because `SEED = 20260825` is fixed in `data.py`. Changing that seed invalidates the frozen
  splits and every number measured against them.
- **The test split is sealed until Phase 4.** `load_split("test")` raises unless `allow_test=True`. Phase 2
  and Phase 3 must never pass that flag.
- **Long runs need `caffeinate -i`.** A laptop sleep mid-run leaves worker threads blocked on half-open
  sockets. The client now has a 120s timeout, but `caffeinate` is still the first line of defence.
- **Use `python -u` when piping output.** Piping through `grep` makes Python block-buffer stdout, so a
  long-running job appears completely silent and progress is invisible.
- Training labels come from the teacher; **all evaluation is against dataset gold.** Both teacher and student
  are scored on the same absolute yardstick, so the student can in principle exceed the teacher.

---

## What We Tried That Failed

| Approach | Why it failed |
|----------|--------------|
| Six-model bake-off, 500 examples, no checkpointing | Ran ~90 min, laptop slept, threads hung on dead sockets with zero CPU. Results were only written after all six models finished, so ~$1 of completed work was lost with nothing saved. Fixed by per-model checkpointing + explicit timeout. |
| Leaving teacher reasoning at its default | Thinking tokens overran `max_tokens=2048`, truncating the JSON. 33.5% unparseable, recall 0.436, micro-F1 read 0.548. The metric was measuring truncation, not capability. |
| `reasoning_effort: "low"` as a middle ground | Made it **worse**, not better — 2,896 output tokens vs 2,459 at default. Only `"none"` works. |
| Taking the first N eligible rows for splits | Source file is ordered by `source_dataset`; its first ~150k rows are 100% Singapore-region. Produced splits that were 83% SG. Fixed with reservoir sampling over the full split. |
| Tightening the `IDCARDNUM` prompt to stop scope leak | Precision rose 0.423 → 0.812 but recall collapsed 0.887 → 0.245; that label's F1 got **worse** (0.573 → 0.377). Kept only because overall precision improved. Prompting is the wrong tool here — see Open Questions. |
| Assuming a flat $0.90/M rate for all >16B models | Wrong by up to 20x. The flat rate only applies to models Fireworks doesn't price individually; real rates span $0.15/$0.60 to $3.00/$15.00. |
| Assuming `Qwen2.5-72B` / `Llama-3.1-70B` were available | Both 404 with `"not deployed"` — retired from serverless. Verified by direct call, not just `models.list()`. They exist only as on-demand deployments at ~$7/hr per H100, rejected as budget- and analysis-distorting. |
| `.gitignore` with `data/` plus `!data/manifest.json` | Git will not descend into an excluded directory, so the negation never applied. Needs `data/*` instead of `data/`. |

---

## Don't Touch

- **`reports/raw/invalid-2048cap/` and `reports/raw/v1-prompt/`** — retained deliberately as evidence of the
  invalidated and superseded runs. Do not move them back into `reports/raw/` or they will be picked up as
  live checkpoints.
- **`data/*.jsonl` and `data/manifest.json`** — frozen. Do not rebuild without deliberately re-freezing and
  recording it, since every measured number is tied to these exact files.
- **`data/test.jsonl`** — sealed until Phase 4. Do not read it, sample from it, or tune anything against it.
- **The `GIVENNAME`/`SURNAME` rules in `SYSTEM`** — already iterated on. The residual ~0.68 F1 is genuine
  annotation ambiguity in the gold data (multicultural names where the given/family boundary is not
  recoverable from the string), not a fixable prompt bug. Further tuning here risks overfitting to `val`.

---

## Next Step

**Implement Phase 2 generation: run the teacher over all 8,000 `train` rows via the Fireworks batch API and
write the training set.** This requires writing the batch path (see Open Questions — it does not exist yet);
the synchronous path in `teacher.py` works but costs double. Validate every output against the schema, drop
failures rather than teaching them, and record the actual dollar cost in `README.md`. Projected $1.88 batched
/ $3.77 synchronous.

---

## Open Questions / Blockers

- **The batch API is not implemented.** `teacher.py` only does synchronous `chat.completions.create`. The
  $1.88 Phase 2 projection assumes Fireworks' 50% batch discount, which needs a separate code path. Decide
  whether to build it or accept $3.77 synchronous — the budget easily absorbs either, so this is a
  time-vs-money call, not a blocker.
- **No GPU decided for the Phase 5 serving benchmark.** Throughput under sustained concurrency needs a
  GPU that stays up; Kaggle sessions are ephemeral and time-limited. Options are a few hours of a rented GPU
  (~$0.30/hr) or rougher in-notebook numbers. Worth settling before Phase 3 finishes.
- **Ceiling was measured at n=200, not 500.** Adequate for choosing between teachers; marginal for a headline
  number. Re-measuring at 500 in Phase 4 costs ~$0.24.
- **`IDCARDNUM` remains at 0.377 F1.** The real fix is restoring `DRIVERLICENSENUM` and `PASSPORTNUM` to the
  label set so the model has somewhere legitimate to put them, but that reopens D-006 and forces a split
  rebuild. Deferred deliberately; revisit if Phase 4 shows the student inheriting the under-detection.

---

## Session History

_Append-only. One line per session — never overwrite previous entries._

- 2026-08-25: Phase 1 complete — task and 12 labels locked, splits frozen (8k/1k/1k, region skew corrected from 83% SG to 12%), metric written and tested, teacher selected by bake-off (`qwen3p7-plus` over `deepseek-v4-pro`, +0.005 F1 for 3.4x cost), ceiling established at 0.832 micro-F1. Found and fixed the reasoning-token truncation bug that had understated the ceiling by 0.28 F1. ~$1.50 spent.

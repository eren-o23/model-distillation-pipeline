# Teacher ceiling

**Teacher:** `accounts/fireworks/models/qwen3p7-plus`
**Measured:** 2026-08-25 on 200 examples from the **val** split.
**The test split has not been touched** — it stays sealed until Phase 4.

## Headline

| metric | value |
|---|---|
| **micro-F1** | **0.832** |
| precision | 0.847 |
| recall | 0.818 |
| schema-invalid | 0/200 (0.0%) |
| hallucinated values | 0 |
| cost | $0.0942 for 200 examples |
| tokens | 125,502 in / 27,525 out |

This is the number every Phase 4 claim is measured against. It is a ceiling **for this teacher**, not for the
task: a frontier model would likely score higher on ambiguous names and mixed-script text.

## Per label

```
label                      P       R      F1      n
DATE                   0.990   0.976   0.983    209
GIVENNAME              0.677   0.683   0.680    199
SURNAME                0.670   0.702   0.686    171
CITY                   0.826   0.820   0.823    133
EMAIL                  1.000   1.000   1.000    131
TELEPHONENUM           0.911   0.981   0.944    104
STREET                 0.885   0.944   0.914     90
ZIPCODE                0.782   0.847   0.813     72
IDCARDNUM              0.812   0.245   0.377     53
CREDITCARDNUMBER       1.000   0.872   0.932     47
TAXNUM                 1.000   0.718   0.836     39
SOCIALNUM              1.000   0.571   0.727     35
--------------------------------------------------
MICRO                  0.847   0.818   0.832   1283

examples: 200  schema-invalid: 0 (0.0%)  hallucinated values: 0
```

Weakest labels: `IDCARDNUM` (0.377), `GIVENNAME` (0.680), `SURNAME` (0.686). These are the classes the student will
find hardest too, so they are worth watching in Phase 3.

## Cost projection

At 628 input and 138 output tokens per example, at $0.40/$1.60 per 1M:

| run | cost |
|---|---|
| Phase 2 generation, 8,000 examples (serverless) | $3.77 |
| Phase 2 generation, 8,000 examples (batch, -50%) | $1.88 |

## Method notes

- 12 PII labels; predictions scored as a multiset of `(label, value)` pairs per label.
- Reasoning is disabled on the teacher (`reasoning_effort: "none"`). Left on, this model spent ~1,900 of
  ~2,459 output tokens thinking, overran the token cap and truncated its own JSON — 33.5% of outputs were
  unparseable and micro-F1 read 0.548 instead of its true value.
- The system prompt encodes the dataset's span conventions (given-name grouping, street excluding the
  building number, verbatim dates, and the in-scope label list). Without these the score understates the
  teacher by measuring annotation mismatch rather than detection.
- Prompt development was done on `val` only.

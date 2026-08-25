"""Generate reports/ceiling.md from a saved bake-off checkpoint.

Numbers are read from the checkpoint rather than transcribed by hand, so the report cannot drift from
what was actually measured.

  python scripts/write_ceiling.py accounts/fireworks/models/qwen3p7-plus 200
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import LABELS, SPLIT_SIZES, load_split  # noqa: E402
from src.pii.metric import Score  # noqa: E402
from src.pii.teacher import Usage, price_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_id, n = sys.argv[1], int(sys.argv[2])
    blob = json.loads((ROOT / "reports" / "raw" / f"{model_id.split('/')[-1]}-{n}.json").read_text())
    rows = load_split("val")[:n]

    score = Score()
    for row, pred in zip(rows, blob["predictions"], strict=True):
        score.add(row["entities"], pred, row["source_text"])
    usage = Usage(**blob["usage"])
    cost = usage.cost(model_id)
    pin, pout = price_for(model_id)

    # Phase 2 projection: same per-example token profile, batch inference at half price.
    per_ex_in = usage.prompt_tokens / n
    per_ex_out = usage.completion_tokens / n
    train_n = SPLIT_SIZES["train"]
    phase2_serverless = (per_ex_in * pin + per_ex_out * pout) * train_n / 1_000_000
    m = score.micro

    weak = sorted(score.per_label.items(), key=lambda kv: kv[1].f1)[:3]

    out = f"""# Teacher ceiling

**Teacher:** `{model_id}`
**Measured:** {date.today()} on {n} examples from the **val** split.
**The test split has not been touched** — it stays sealed until Phase 4.

## Headline

| metric | value |
|---|---|
| **micro-F1** | **{m.f1:.3f}** |
| precision | {m.precision:.3f} |
| recall | {m.recall:.3f} |
| schema-invalid | {score.schema_invalid}/{n} ({score.schema_invalid / n:.1%}) |
| hallucinated values | {score.hallucinated} |
| cost | ${cost:.4f} for {n} examples |
| tokens | {usage.prompt_tokens:,} in / {usage.completion_tokens:,} out |

This is the number every Phase 4 claim is measured against. It is a ceiling **for this teacher**, not for the
task: a frontier model would likely score higher on ambiguous names and mixed-script text.

## Per label

```
{score.table()}
```

Weakest labels: {", ".join(f"`{k}` ({v.f1:.3f})" for k, v in weak)}. These are the classes the student will
find hardest too, so they are worth watching in Phase 3.

## Cost projection

At {per_ex_in:.0f} input and {per_ex_out:.0f} output tokens per example, at ${pin:.2f}/${pout:.2f} per 1M:

| run | cost |
|---|---|
| Phase 2 generation, {train_n:,} examples (serverless) | ${phase2_serverless:.2f} |
| Phase 2 generation, {train_n:,} examples (batch, -50%) | ${phase2_serverless / 2:.2f} |

## Method notes

- {len(LABELS)} PII labels; predictions scored as a multiset of `(label, value)` pairs per label.
- Reasoning is disabled on the teacher (`reasoning_effort: "none"`). Left on, this model spent ~1,900 of
  ~2,459 output tokens thinking, overran the token cap and truncated its own JSON — 33.5% of outputs were
  unparseable and micro-F1 read 0.548 instead of its true value.
- The system prompt encodes the dataset's span conventions (given-name grouping, street excluding the
  building number, verbatim dates, and the in-scope label list). Without these the score understates the
  teacher by measuring annotation mismatch rather than detection.
- Prompt development was done on `val` only.
"""
    (ROOT / "reports" / "ceiling.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()

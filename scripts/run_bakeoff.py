"""Teacher bake-off: score candidate models on the same val examples, report F1 and actual cost.

Picks the teacher on evidence rather than on model size. Spending requires an explicit --yes.

  python scripts/run_bakeoff.py --models <id1>,<id2> --limit 20
  python scripts/run_bakeoff.py --models <id1>,<id2> --limit 300 --yes
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import load_split  # noqa: E402
from src.pii.metric import Score  # noqa: E402
from src.pii.teacher import SYSTEM, Usage, client, extract, price_for  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def estimate_cost(rows: list[dict], model_id: str) -> float:
    """Rough pre-flight estimate: ~4 chars per token in, ~120 tokens out per example."""
    pin, pout = price_for(model_id)
    prompt_tokens = sum(len(SYSTEM) + len(r["source_text"]) for r in rows) / 4
    return (prompt_tokens * pin + 120 * len(rows) * pout) / 1_000_000


def run_model(cli, model_id: str, rows: list[dict], workers: int) -> tuple[Score, Usage]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda r: extract(cli, model_id, r["source_text"]), rows))

    score, total = Score(), Usage()
    for row, (entities, usage) in zip(rows, results, strict=True):
        score.add(row["entities"], entities, row["source_text"])
        total.prompt_tokens += usage.prompt_tokens
        total.completion_tokens += usage.completion_tokens
    return score, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated Fireworks model IDs")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true", help="required to actually spend credits")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = load_split("val")[: args.limit]  # val only — never test (sealed until Phase 4)

    total_estimate = sum(estimate_cost(rows, m) for m in models)
    print(f"{len(models)} model(s) x {len(rows)} val examples")
    for m in models:
        print(f"  {m:<62} ~${estimate_cost(rows, m):.2f}")
    print(f"  estimated total: ~${total_estimate:.2f}")

    if not args.yes:
        print("\nDry run. Re-run with --yes to spend credits.")
        return

    cli = client()
    results = {}
    for model_id in models:
        print(f"\nrunning {model_id} …")
        score, usage = run_model(cli, model_id, rows, args.workers)
        cost = usage.cost(model_id)
        results[model_id] = (score, usage, cost)
        print(score.table())
        print(f"  cost: ${cost:.4f}  ({usage.prompt_tokens} in / {usage.completion_tokens} out)")

    REPORTS.mkdir(exist_ok=True)
    lines = [
        "# Teacher bake-off",
        "",
        f"Date: {date.today()}  ·  {len(rows)} examples from the **val** split (test remains sealed).",
        "",
        "| model | micro-F1 | P | R | schema-invalid | hallucinated | cost | F1 per $ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model_id, (score, _usage, cost) in results.items():
        m = score.micro
        lines.append(
            f"| `{model_id}` | **{m.f1:.3f}** | {m.precision:.3f} | {m.recall:.3f} | "
            f"{score.schema_invalid / len(rows):.1%} | {score.hallucinated} | "
            f"${cost:.4f} | {m.f1 / cost if cost else 0:.0f} |"
        )

    for model_id, (score, _u, _c) in results.items():
        lines += ["", f"## {model_id}", "", "```", score.table(), "```"]

    total_spend = sum(c for _s, _u, c in results.values())
    lines += ["", f"**Total spend: ${total_spend:.4f}**"]
    (REPORTS / "bakeoff.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote reports/bakeoff.md  ·  total spend ${total_spend:.4f}")


if __name__ == "__main__":
    main()

"""List Fireworks serverless models, so bake-off model IDs come from the live catalogue, not a guess."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.teacher import client, price_for  # noqa: E402

if __name__ == "__main__":
    needle = sys.argv[1].lower() if len(sys.argv) > 1 else "qwen"
    models = sorted(m.id for m in client().models.list().data if needle in m.id.lower())
    if not models:
        print(f"No models matching {needle!r}.")
        sys.exit(1)
    print(f"{len(models)} model(s) matching {needle!r}:  [$in/$out per 1M tokens]")
    for m in models:
        pin, pout = price_for(m)
        print(f"  {m:<62} {pin:>5.2f} / {pout:<5.2f}")

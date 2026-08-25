"""Build and freeze the train/val/test splits. Run once; the manifest pins what was built."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import DATA_DIR, LABELS, build_splits  # noqa: E402

if __name__ == "__main__":
    print(f"Building splits over {len(LABELS)} labels -> {DATA_DIR}")
    manifest = build_splits()

    for name, info in manifest["splits"].items():
        print(f"\n{name}: {info['n']} rows, {info['entities']} entities  sha256={info['sha256'][:12]}…")
        print(f"  regions: {info['regions']}")
        for label, count in info["label_counts"].items():
            print(f"    {label:<20}{count:>7}{count / info['n']:>8.2f}/row")

    # Splits must be disjoint by uid — the whole benchmark rests on it.
    uids = {
        name: {r["uid"] for r in map(json.loads, (DATA_DIR / f"{name}.jsonl").read_text().splitlines())}
        for name in manifest["splits"]
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = uids[a] & uids[b]
        assert not overlap, f"{a}/{b} overlap: {len(overlap)} uids"
    print("\nuid disjointness: OK")

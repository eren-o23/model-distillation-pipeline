"""Dataset loading, label scope, and frozen train/val/test splits.

Source: ai4privacy/pii-masking-openpii-1.5m (CC-BY-4.0). See docs/decisions.md D-002.

The 12 labels below were chosen by measuring English label frequency across a 60k-row sample and then
selecting for redaction value among the classes frequent enough to learn (D-006). A pure frequency cut
would have kept TITLE ("Mr", "Master") while dropping CREDITCARDNUMBER and SOCIALNUM — a redactor that
masks honorifics but leaks card numbers.
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATASET = "ai4privacy/pii-masking-openpii-1.5m"
LANGUAGE = "en"
SEED = 20260825

LABELS = frozenset(
    {
        "GIVENNAME",
        "SURNAME",
        "DATE",
        "EMAIL",
        "CITY",
        "TELEPHONENUM",
        "STREET",
        "ZIPCODE",
        "IDCARDNUM",
        "CREDITCARDNUMBER",
        "TAXNUM",
        "SOCIALNUM",
    }
)

SPLIT_SIZES = {"train": 8000, "val": 1000, "test": 1000}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Example:
    uid: str
    source_text: str
    entities: list[dict[str, str]]
    region: str


def gold_entities(row: dict) -> list[dict[str, str]]:
    """Extract in-scope gold entities as {label, value}, discarding offsets (D-003)."""
    return [
        {"label": m["label"], "value": m["value"]}
        for m in row["privacy_mask"]
        if m["label"] in LABELS
    ]


def template_key(row: dict) -> str:
    """Fingerprint a row with its PII values blanked out, so two rows built from the same carrier
    sentence collapse to one key even though their names and numbers differ.

    The split build already drops exact-duplicate text; this catches the templated near-duplicates the
    spec asks about. Values are masked longest-first so a substring value can't corrupt a longer one.
    Measured on the frozen train split: 8 collisions in 8,000 rows, so this is a cheap confirmation
    rather than a meaningful reduction.
    """
    text = row["source_text"]
    for e in sorted(row["entities"], key=lambda e: -len(e["value"])):
        text = text.replace(e["value"], "\0")
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def _collect(hf_split: str, n_needed: int, seen: set[str]) -> list[Example]:
    """Stream all of `hf_split` and reservoir-sample n_needed English rows carrying in-scope entities.

    Reservoir sampling rather than taking the first N: the source file is ordered by source_dataset,
    so the opening ~150k rows are entirely Singapore-region and only then do CA/GB/US/IN appear. Any
    early-stopping strategy yields an ~83% SG sample. A reservoir gives a uniform draw over the whole
    split at O(n_needed) memory, at the cost of one full pass.
    """
    from datasets import load_dataset

    rng = random.Random(SEED)
    ds = load_dataset(DATASET, split=hf_split, streaming=True)
    reservoir: list[Example] = []
    n_eligible = 0

    for row in ds:
        if row["language"] != LANGUAGE:
            continue
        ents = gold_entities(row)
        if not ents:
            continue  # every example must carry signal
        fingerprint = hashlib.sha256(row["source_text"].encode()).hexdigest()
        if fingerprint in seen:
            continue  # exact-duplicate text, within or across splits, would leak
        seen.add(fingerprint)

        ex = Example(str(row["uid"]), row["source_text"], ents, row.get("region", "?"))
        if len(reservoir) < n_needed:
            reservoir.append(ex)
        else:
            j = rng.randrange(n_eligible + 1)
            if j < n_needed:
                reservoir[j] = ex
        n_eligible += 1

    if len(reservoir) < n_needed:
        raise RuntimeError(f"{hf_split}: only {len(reservoir)} usable rows, need {n_needed}")

    print(f"  {hf_split}: sampled {n_needed} from {n_eligible} eligible English rows")
    rng.shuffle(reservoir)
    return reservoir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_splits(out_dir: Path = DATA_DIR) -> dict:
    """Build and freeze all three splits. train comes from the HF train split, val/test from
    validation, so the three cannot overlap by construction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    train = _collect("train", SPLIT_SIZES["train"], seen)
    holdout = _collect("validation", SPLIT_SIZES["val"] + SPLIT_SIZES["test"], seen)
    val, test = holdout[: SPLIT_SIZES["val"]], holdout[SPLIT_SIZES["val"] :]

    manifest = {"dataset": DATASET, "language": LANGUAGE, "seed": SEED, "labels": sorted(LABELS), "splits": {}}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps({"uid": r.uid, "source_text": r.source_text, "entities": r.entities}) + "\n")
        labels = Counter(e["label"] for r in rows for e in r.entities)
        manifest["splits"][name] = {
            "n": len(rows),
            "sha256": _sha256(path),
            "entities": sum(labels.values()),
            "label_counts": dict(labels.most_common()),
            "regions": dict(Counter(r.region for r in rows).most_common(8)),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(name: str, allow_test: bool = False, data_dir: Path = DATA_DIR) -> list[dict]:
    """Load a frozen split.

    The test set stays sealed until Phase 4; loading it takes an explicit opt-in so it cannot be
    picked up by accident during data generation or training.
    """
    if name == "test" and not allow_test:
        raise RuntimeError(
            "The test set is sealed until Phase 4 (benchmarking). "
            "Pass allow_test=True only from the final benchmark."
        )
    path = data_dir / f"{name}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f]

"""Scoring for PII entity extraction.

Gold and predictions are both lists of {"label": str, "value": str}. Entities are compared as a
*multiset per label*: a text containing two GIVENNAMEs requires two correct predictions. Set-based
comparison would silently reward under-prediction, which is exactly the failure mode that matters
for redaction.
"""

import random
from collections import Counter
from dataclasses import dataclass, field

Entity = dict[str, str]


def normalise(value: str) -> str:
    """Collapse whitespace and casefold. Matching PII values shouldn't hinge on spacing or case."""
    return " ".join(value.split()).casefold()


def _key(entities: list[Entity]) -> Counter:
    return Counter((e["label"], normalise(e["value"])) for e in entities)


def is_hallucinated(entity: Entity, source_text: str) -> bool:
    """True if the value does not actually occur in the input.

    Scoring counts these; Phase 2 generation strips them from the training set. Same predicate either
    way, defined once: a value absent from the text can never be a correct extraction, so teaching it
    would teach the student to invent.
    """
    return normalise(entity["value"]) not in normalise(source_text)


def strip_hallucinated(entities: list[Entity], source_text: str) -> list[Entity]:
    """Drop only the invented entities, keeping the rest of the example's labels."""
    return [e for e in entities if not is_hallucinated(e, source_text)]


@dataclass
class LabelScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Score:
    micro: LabelScore = field(default_factory=LabelScore)
    per_label: dict[str, LabelScore] = field(default_factory=dict)
    n_examples: int = 0
    # Phase 5's router escalates on exactly these two conditions, so we measure them from the start.
    schema_invalid: int = 0
    hallucinated: int = 0

    def add(self, gold: list[Entity], pred: list[Entity] | None, source_text: str = "") -> None:
        """Score one example. pred=None means the model's output failed schema validation."""
        self.n_examples += 1

        if pred is None:
            self.schema_invalid += 1
            pred = []  # unparseable output detects nothing; every gold entity is a miss

        if source_text:
            self.hallucinated += sum(1 for e in pred if is_hallucinated(e, source_text))

        g, p = _key(gold), _key(pred)
        for label in {lbl for lbl, _ in g | p}:
            gl = Counter({k: v for k, v in g.items() if k[0] == label})
            pl = Counter({k: v for k, v in p.items() if k[0] == label})
            hit = gl & pl
            tp, fp, fn = sum(hit.values()), sum((pl - gl).values()), sum((gl - pl).values())

            s = self.per_label.setdefault(label, LabelScore())
            s.tp, s.fp, s.fn = s.tp + tp, s.fp + fp, s.fn + fn
            self.micro.tp += tp
            self.micro.fp += fp
            self.micro.fn += fn

    def table(self) -> str:
        rows = [f"{'label':<20} {'P':>7} {'R':>7} {'F1':>7} {'n':>6}"]
        for label in sorted(self.per_label, key=lambda k: -(self.per_label[k].tp + self.per_label[k].fn)):
            s = self.per_label[label]
            rows.append(
                f"{label:<20} {s.precision:>7.3f} {s.recall:>7.3f} {s.f1:>7.3f} {s.tp + s.fn:>6}"
            )
        m = self.micro
        rows.append(f"{'-' * 50}")
        rows.append(f"{'MICRO':<20} {m.precision:>7.3f} {m.recall:>7.3f} {m.f1:>7.3f} {m.tp + m.fn:>6}")
        rows.append(
            f"\nexamples: {self.n_examples}  "
            f"schema-invalid: {self.schema_invalid} ({self.schema_invalid / max(self.n_examples, 1):.1%})  "
            f"hallucinated values: {self.hallucinated}"
        )
        return "\n".join(rows)


def score(
    golds: list[list[Entity]], preds: list[list[Entity] | None], sources: list[str] | None = None
) -> Score:
    sources = sources or [""] * len(golds)
    s = Score()
    for gold, pred, src in zip(golds, preds, sources, strict=True):
        s.add(gold, pred, src)
    return s


def row_counts(golds: list[list[Entity]], preds: list[list[Entity] | None]) -> list[tuple[int, int, int]]:
    """Per-row (tp, fp, fn), scored by the same rules as `score()`.

    Micro-F1 is a ratio of sums over rows, so keeping the counts per row makes a bootstrap resample a
    matter of adding integers rather than re-scoring every example on every draw.
    """
    out = []
    for gold, pred in zip(golds, preds, strict=True):
        s = Score()
        s.add(gold, pred)
        out.append((s.micro.tp, s.micro.fp, s.micro.fn))
    return out


def micro_f1(counts: list[tuple[int, int, int]], idx: list[int] | None = None) -> float:
    """Micro-F1 over a selection of per-row counts; `idx` may repeat rows, as a bootstrap draw does."""
    tp = fp = fn = 0
    for i in range(len(counts)) if idx is None else idx:
        a, b, c = counts[i]
        tp, fp, fn = tp + a, fp + b, fn + c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def bootstrap_delta(a: list[tuple[int, int, int]], b: list[tuple[int, int, int]], draws: int, seed: int
                    ) -> tuple[float, float, float]:
    """95% CI on micro_f1(a) - micro_f1(b), resampling rows with replacement.

    Paired: each draw scores both models on the same resampled rows. That is only meaningful once both
    have been measured on one split at one n, and it is what makes the interval tight enough to decide
    "matches" against "beats" — the same rows are hard for both models, and resampling them together
    cancels that shared difficulty instead of counting it as evidence twice.

    Returns (low, high, fraction of draws where a > b).
    """
    rng = random.Random(seed)
    n = len(a)
    deltas = sorted(
        micro_f1(a, idx) - micro_f1(b, idx)
        for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(draws))
    )
    return deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1], sum(d > 0 for d in deltas) / draws

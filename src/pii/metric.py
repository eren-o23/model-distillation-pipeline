"""Scoring for PII entity extraction.

Gold and predictions are both lists of {"label": str, "value": str}. Entities are compared as a
*multiset per label*: a text containing two GIVENNAMEs requires two correct predictions. Set-based
comparison would silently reward under-prediction, which is exactly the failure mode that matters
for redaction.
"""

from collections import Counter
from dataclasses import dataclass, field

Entity = dict[str, str]


def normalise(value: str) -> str:
    """Collapse whitespace and casefold. Matching PII values shouldn't hinge on spacing or case."""
    return " ".join(value.split()).casefold()


def _key(entities: list[Entity]) -> Counter:
    return Counter((e["label"], normalise(e["value"])) for e in entities)


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
            self.hallucinated += sum(
                1 for e in pred if normalise(e["value"]) not in normalise(source_text)
            )

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

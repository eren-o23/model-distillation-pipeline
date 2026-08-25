"""Prompt format for the fine-tuned student.

Phase 3 trains on this and Phase 5 serves with it. They must agree exactly — a student trained on one
system prompt and served with another silently loses accuracy, and the loss looks like a bad fine-tune.
Import from here in both places rather than re-typing the string.

The prompt is deliberately much shorter than `teacher.SYSTEM` (D-017). The teacher needs ~475 tokens of
annotation conventions spelled out; the student learns them from the weights, so at serving time it only
needs to be told what shape to emit. That cuts input tokens per request roughly 4x, which raises vLLM
throughput and lowers the break-even volume — the number this project exists to produce.
"""

import json

from src.pii.data import LABELS

SHORT_SYSTEM = (
    "Extract PII spans as JSON. Labels: "
    + ", ".join(sorted(LABELS))
    + '\nReturn only: {"entities": [{"label": "...", "value": "..."}]}'
)


def chat_messages(source_text: str) -> list[dict[str, str]]:
    """The prompt half of a training example, and the whole request at serving time."""
    return [
        {"role": "system", "content": SHORT_SYSTEM},
        {"role": "user", "content": source_text},
    ]


def to_sft_example(source_text: str, entities: list[dict[str, str]]) -> dict:
    """One training row. The assistant turn is compact JSON: the student is trained to emit exactly the
    bytes it should emit at inference, so no whitespace it would have to reproduce is added here."""
    answer = json.dumps({"entities": entities}, separators=(",", ":"), ensure_ascii=False)
    return {"messages": chat_messages(source_text) + [{"role": "assistant", "content": answer}]}

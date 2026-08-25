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


def render_prompt(tokenizer, source_text: str) -> str:
    """The exact string fed to the model at inference — and the exact prefix it was trained on.

    `enable_thinking=False` is load-bearing, and its effect is the opposite of what the name suggests.
    Qwen3 is a hybrid reasoning model whose template handles the three cases inconsistently:

        add_generation_prompt=True, flag unset      -> ends at "<|im_start|>assistant\\n"
        add_generation_prompt=True, thinking=False  -> appends "<think>\\n\\n</think>\\n\\n"
        a full conversation with an assistant turn  -> ALWAYS appends "<think>\\n\\n</think>\\n\\n"

    Training renders the third form, so an empty think block sits in front of every training answer.
    Serving with the flag unset would hand the model a prefix it has never seen — no error, no warning,
    just a student that reads as a bad fine-tune. Passing enable_thinking=False here is what makes the two
    agree, which is why train and serve must both come through this function rather than calling
    apply_chat_template themselves. The identity is asserted in tests/test_prompt_render.py.
    """
    return tokenizer.apply_chat_template(
        chat_messages(source_text),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_sft(tokenizer, row: dict) -> tuple[str, str]:
    """Split one training row into (prompt, completion) — the boundary the loss mask runs along.

    Returned as two strings rather than one joined string because the trainer tokenises them separately,
    so `labels` can be -100 across the prompt and only the answer contributes loss. That split is only
    safe if tokenising the halves gives the same ids as tokenising the whole; measured over 500 rows it
    is token-for-token identical, and a test pins it so a tokenizer change cannot quietly break it.

    The completion ends at EOS deliberately. The template's own trailing newline after <|im_end|> is
    dropped: generation stops at EOS, so training the model to emit anything after it teaches a token it
    will never get to use.
    """
    _, user, assistant = row["messages"]
    return render_prompt(tokenizer, user["content"]), assistant["content"] + tokenizer.eos_token

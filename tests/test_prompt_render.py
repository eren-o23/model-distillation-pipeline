"""Guards on the student's prompt rendering.

Every failure these catch is silent. A student trained on one prefix and served with another does not
raise — it just scores badly, and the loss looks like a bad fine-tune rather than a plumbing bug. Qwen3's
chat template makes that specific mistake easy to reach from the default code path (see
`student.render_prompt`), so the train/serve identity is asserted here rather than assumed.

Needs the real Qwen3 tokenizer — a ~11MB download on first run, cached afterwards. No torch, no GPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.student import SHORT_SYSTEM, render_prompt, render_sft, to_sft_example  # noqa: E402
from src.pii.teacher import _parse  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
ENTITIES = [{"label": "GIVENNAME", "value": "Ana"}, {"label": "CITY", "value": "Košice"}]
TEXT = "Ana lives in Košice"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers", reason="tokenizer-only; venv stays torch-free")
    pytest.importorskip("jinja2", reason="apply_chat_template needs jinja2")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def row():
    return to_sft_example(TEXT, ENTITIES)


def test_prompt_carries_the_system_prompt_verbatim(tok):
    """SHORT_SYSTEM is what D-017's 4x token saving is measured against; the template must not reflow it."""
    assert SHORT_SYSTEM in render_prompt(tok, TEXT)


def test_prompt_ends_with_a_closed_empty_think_block(tok):
    """Pins which of Qwen3's three template branches we are on.

    An *open* <think> would mean the model is being invited to reason before answering — the same class of
    bug as D-013 on the teacher, where reasoning tokens overran the cap and truncated the JSON.
    """
    prompt = render_prompt(tok, TEXT)
    assert prompt.endswith("<think>\n\n</think>\n\n"), prompt[-40:]
    assert prompt.count("<think>") == prompt.count("</think>") == 1


def test_training_target_is_exactly_what_serving_will_ask_for(tok, row):
    """The invariant the whole phase rests on: prompt + completion reconstructs the training render."""
    prompt, completion = render_sft(tok, row)
    full = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    assert full.startswith(prompt), "serving prefix is not a prefix of the training render"
    assert full[len(prompt) :].rstrip("\n") == completion


def test_splitting_at_the_mask_boundary_does_not_move_a_single_token(tok, row):
    """Label masking tokenises the halves separately. If the tokenizer merged across the boundary, the
    model would train on ids it can never be given at inference — and nothing would report it."""
    prompt, completion = render_sft(tok, row)

    def ids(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    assert ids(prompt) + ids(completion) == ids(prompt + completion)


def test_completion_still_parses_with_the_scorer_the_student_is_graded_by(tok, row):
    """Closes the loop with tests/test_sft.py: the bytes after the mask boundary, minus EOS, must survive
    the same validator Phase 4 and Phase 5 will score the student's real output with."""
    _, completion = render_sft(tok, row)
    assert completion.endswith(tok.eos_token)
    assert _parse(completion.removesuffix(tok.eos_token)) == ENTITIES


def test_pad_and_eos_are_different_tokens(tok):
    """Recipes routinely set `pad_token = eos_token`. Qwen3 ships a distinct pad token, and keeping it
    distinct is what lets the collator pad a batch without the model reading padding as a stop signal."""
    assert tok.pad_token_id is not None
    assert tok.pad_token_id != tok.eos_token_id

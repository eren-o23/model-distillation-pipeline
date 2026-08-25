"""Generate with the student and score it against gold.

Used three ways, all through `evaluate()`: the Phase 3 baselines (untuned Qwen3-8B), the per-epoch
validation inside training, and Phase 4's benchmark on the sealed test set. One code path so those three
numbers are comparable — an eval that differs between phases produces a delta that measures the harness.

GPU-side module: importing it requires torch. Nothing in the local test suite imports it.
"""

import os

import torch

from src.pii.metric import Score
from src.pii.student import SHORT_SYSTEM, render_prompt
from src.pii.teacher import _parse

BASE_MODEL = "Qwen/Qwen3-8B"

# The training answers reach 398 tokens at their longest (measured on train_sft.jsonl with this exact
# tokenizer; p99 is 246). 512 leaves headroom without paying for it on every sequence. An untuned base
# model that rambles past the cap gets truncated and scores as schema-invalid, which is the honest
# reading of a model that cannot stop.
MAX_NEW_TOKENS = 512


def load_model(adapter: str | None = None, base: str = BASE_MODEL):
    """Load the 4-bit base, optionally with a LoRA adapter on top. Returns (model, tokenizer).

    fp16 compute, not bf16: Kaggle's T4s are compute capability 7.5 and bf16 needs Ampere. The same
    constraint rules out FlashAttention-2, so attention falls back to SDPA (D-019).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Pin to this process's GPU rather than sharding across both. A 4-bit 8B is ~5.5GB and fits one 16GB
    # T4, so under torchrun each rank owns a full replica and the two GPUs do real data parallelism;
    # device_map="auto" would instead split one model across both and leave each idle half the time.
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        dtype=torch.float16,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer(base)


def tokenizer(base: str = BASE_MODEL):
    """Left-padded, because generation is the only thing this module does.

    With the default right padding, a batched `generate` continues from padding rather than from the last
    real token. It does not raise — it quietly produces worse output, which would read as the student
    being weaker than it is.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    return tok


@torch.no_grad()
def generate(model, tok, prompts: list[str], batch_size: int = 8) -> list[str]:
    """Greedy batched generation, length-sorted.

    Sorting by prompt length before batching, then restoring the original order, keeps short prompts out
    of the same padded batch as the 871-token tail. Padding is charged at the longest member of each
    batch, so on this length distribution (mean 276, max 871) the sort is most of the eval's throughput.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: dict[int, str] = {}

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: the benchmark must be reproducible run to run
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        # Slice off the prompt by width rather than by string matching: left padding makes every row in
        # the batch start at the same offset, so this is exact.
        completions = tok.batch_decode(gen[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        out.update(zip(idx, completions, strict=True))

    return [out[i] for i in range(len(prompts))]


def evaluate(model, tok, rows: list[dict], system: str = SHORT_SYSTEM, batch_size: int = 8):
    """Score the model over `rows` (frozen-split format: uid, source_text, entities).

    Scoring is against dataset **gold**, never against the teacher — the same absolute yardstick used for
    the teacher ceiling, which is what lets the student in principle exceed it.

    Returns (Score, samples) where samples is a handful of (source, gold, raw, parsed) for eyeballing in
    W&B. `_parse` is the teacher's validator, reused deliberately: the student is graded by exactly the
    predicate Phase 5's router will escalate on.
    """
    prompts = [render_prompt(tok, r["source_text"], system) for r in rows]
    raws = generate(model, tok, prompts, batch_size)

    s = Score()
    for row, raw in zip(rows, raws, strict=True):
        s.add(row["entities"], _parse(raw), row["source_text"])

    samples = [
        {
            "source_text": r["source_text"],
            "gold": r["entities"],
            "raw": raw,
            "parsed": _parse(raw),
        }
        for r, raw in list(zip(rows, raws, strict=True))[:10]
    ]
    return s, samples


def as_dict(s: Score) -> dict:
    """Flatten a Score for W&B and for reports/raw/phase3/*.json.

    Every number in reports/phase3.md is generated from these files rather than transcribed, the same
    discipline write_ceiling.py uses — a report cannot drift from what was measured if nobody types it.
    """
    return {
        "micro_f1": s.micro.f1,
        "micro_precision": s.micro.precision,
        "micro_recall": s.micro.recall,
        "n_examples": s.n_examples,
        "schema_invalid": s.schema_invalid,
        "schema_invalid_rate": s.schema_invalid / max(s.n_examples, 1),
        "hallucinated": s.hallucinated,
        "per_label": {
            k: {"precision": v.precision, "recall": v.recall, "f1": v.f1, "support": v.tp + v.fn}
            for k, v in sorted(s.per_label.items())
        },
    }

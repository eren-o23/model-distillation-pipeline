"""Teacher client for Fireworks (OpenAI-compatible endpoint).

Model IDs are NOT hardcoded — Fireworks IDs look like `accounts/fireworks/models/<slug>` and a wrong
slug only fails at call time. Use `scripts/list_models.py` to resolve real IDs from the live catalogue.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.pii.data import LABELS

BASE_URL = "https://api.fireworks.ai/inference/v1"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

# $ per 1M tokens (input, output), from docs.fireworks.ai/serverless/pricing (checked 2026-08-25).
# Models Fireworks doesn't price individually fall back to the flat >16B-param rate of $0.90 both ways.
# Batch inference is half these rates. Rates move — re-check before trusting a cost estimate.
PRICING = {
    "deepseek-v4-pro": (1.74, 3.48),
    "deepseek-v4-flash": (0.22, 0.66),
    "glm-5p2": (1.40, 4.40),
    "kimi-k3": (3.00, 15.00),
    "kimi-k2p6": (0.95, 4.00),
    "minimax-m3": (0.30, 1.20),
    "qwen3p7-plus": (0.40, 1.60),
    "qwen3p8": (2.00, 6.00),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.07, 0.30),
}
FLAT_RATE = (0.90, 0.90)


def price_for(model_id: str) -> tuple[float, float]:
    """Longest-match wins, so 'kimi-k2p6' isn't shadowed by a shorter key."""
    matches = [(k, v) for k, v in PRICING.items() if k in model_id.lower()]
    return max(matches, key=lambda kv: len(kv[0]))[1] if matches else FLAT_RATE


class Entity(BaseModel):
    label: str
    value: str


class Extraction(BaseModel):
    entities: list[Entity]


SCHEMA = Extraction.model_json_schema()

SYSTEM = f"""You extract personally identifiable information (PII) from text.

Return every PII span you find, using ONLY these labels:
{", ".join(sorted(LABELS))}

Rules:
- Copy each value EXACTLY as it appears in the text, character for character. Never paraphrase, reformat,
  normalise, shorten or correct it. If a date is written "2016-11-25T00:00:00", return the whole string
  including the time part — do not trim it to "2016-11-25".
- Report each occurrence separately. If the same value appears twice, return it twice.
- GIVENNAME covers ALL of a person's given/first/middle names as a SINGLE entity. For "Dacian Cosmin Ionescu"
  return GIVENNAME "Dacian Cosmin" and SURNAME "Ionescu" — never split the given names into two entities.
- SURNAME is the family name only.
- STREET is the street name WITHOUT the building or house number. For "1222 Chanditala Road" the STREET is
  "Chanditala Road". Never include the leading number.
- CITY is the city name alone, kept separate from STREET.
- DATE covers dates of birth and any other calendar date.
- If the text contains no PII, return an empty list.
- Return ONLY JSON matching: {{"entities": [{{"label": "...", "value": "..."}}]}}"""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def cost(self, model_id: str) -> float:
        pin, pout = price_for(model_id)
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000


def load_key() -> str:
    """Read FIREWORKS_API_KEY from the environment, falling back to .env (which is gitignored)."""
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise RuntimeError(
            "FIREWORKS_API_KEY not set. Add it to .env (gitignored):\n"
            '  echo "FIREWORKS_API_KEY=fw_..." > .env'
        )
    return key


def client(timeout: float = 120.0, max_retries: int = 2):
    """Explicit timeout: the SDK default is long enough that a laptop sleeping mid-run leaves worker
    threads blocked on half-open sockets for many minutes before anything errors."""
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=load_key(), timeout=timeout, max_retries=max_retries)


def _parse(text: str) -> list[dict[str, str]] | None:
    """Validate model output. Returns None on anything unusable — the caller counts that as
    schema-invalid, which is one of Phase 5's escalation triggers."""
    if not text:
        return None
    # Models sometimes wrap JSON in prose or a fenced block; recover the outermost object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = Extraction.model_validate_json(match.group(0))
    except (ValidationError, json.JSONDecodeError):
        return None
    # Labels outside the task's scope are dropped rather than scored as errors: the model volunteering
    # AGE when AGE is out of scope is not a detection failure.
    return [{"label": e.label, "value": e.value} for e in parsed.entities if e.label in LABELS]


def extract(cli, model_id: str, source_text: str, retries: int = 1) -> tuple[list[dict] | None, Usage]:
    """Extract PII from one text. Returns (entities|None, usage); None means schema-invalid."""
    usage = Usage()
    kwargs = {
        "model": model_id,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": source_text}],
        "temperature": 0,
        # Headroom: the answer is ~300 tokens, but a model that ignores reasoning_effort needs room to
        # finish rather than truncating mid-JSON and being scored as a detection failure.
        "max_tokens": 4096,
        "response_format": {"type": "json_object", "schema": SCHEMA},
        # PII extraction is a bounded lookup, not a reasoning problem. Left on, qwen3p7-plus spends ~1,900
        # of its ~2,459 output tokens thinking — billed as output, ~8.7x the cost, and enough to overrun a
        # 2048 cap and truncate the answer. 'none' cuts output to ~282 tokens. ('low' makes it worse, not
        # better.) Ignored by models that don't support it.
        "reasoning_effort": "none",
    }

    for attempt in range(retries + 1):
        try:
            resp = cli.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - one retry, then record as invalid
            # Models that don't know reasoning_effort reject the whole request; drop it and retry once
            # rather than failing every call against that model.
            if "reasoning_effort" in str(exc) and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                continue
            if attempt == retries:
                print(f"    api error: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                return None, usage
            continue

        if resp.usage:
            usage.prompt_tokens += resp.usage.prompt_tokens
            usage.completion_tokens += resp.usage.completion_tokens

        entities = _parse(resp.choices[0].message.content or "")
        if entities is not None:
            return entities, usage

    return None, usage

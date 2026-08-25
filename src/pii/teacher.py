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
- Copy each value EXACTLY as it appears in the text. Never paraphrase, reformat, or correct it.
- Report each occurrence separately. If the same value appears twice, return it twice.
- Use GIVENNAME for first/given names and SURNAME for family names, as separate entities.
- DATE covers dates of birth and any other calendar date. Do not include times.
- STREET is the street name and CITY the city; keep them as separate entities.
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


def client():
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=load_key())


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
        "max_tokens": 2048,
        "response_format": {"type": "json_object", "schema": SCHEMA},
    }

    for attempt in range(retries + 1):
        try:
            resp = cli.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - one retry, then record as invalid
            if attempt == retries:
                print(f"    api error: {type(exc).__name__}: {str(exc)[:120]}")
                return None, usage
            continue

        if resp.usage:
            usage.prompt_tokens += resp.usage.prompt_tokens
            usage.completion_tokens += resp.usage.completion_tokens

        entities = _parse(resp.choices[0].message.content or "")
        if entities is not None:
            return entities, usage

    return None, usage

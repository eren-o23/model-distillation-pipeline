"""The escalation router: the student answers, the teacher is called when the student is unsure.

Two triggers, and they are not equals:

- **schema-invalid** — `_parse()` returned None. A hard trigger: an unparseable answer detects nothing,
  so escalating costs one API call and recovers a whole row. Phase 4 measured this at 0/1000 on both
  models, so it is a safety net rather than a policy, and the report says so rather than claiming a
  validation-driven router that never actually fires.
- **low confidence** — the completion's logprobs, against a threshold swept on `val`. This is the
  policy, and it is what makes the escalation rate a number the operator can choose rather than one the
  model happens to produce.

The student is graded by exactly the predicate this escalates on: `src/pii/eval.py` scores with
`teacher._parse`, so "schema-invalid" means the same thing in the benchmark and in the router.

Serving goes through `/v1/completions` with the string `student.render_prompt` produces, NOT through
`/v1/chat/completions`. Qwen3's chat template appends `<think>\\n\\n</think>\\n\\n` in some cases and not
others, and letting the server apply it hands the model a prefix it was never trained on — no error, no
warning, just a student that reads as a bad fine-tune. That is D-013's failure mode and it has already
cost this project twice. Rendering locally removes the whole class of bug.
"""

import time
from dataclasses import dataclass

from src.pii.student import SHORT_SYSTEM, render_prompt
from src.pii.teacher import _parse

BASE_MODEL = "Qwen/Qwen3-8B"
VLLM_URL = "http://localhost:8000/v1"
# Matches eval.MAX_NEW_TOKENS. The benchmark and the router must cap generation identically or the
# router's schema-invalid rate measures a different truncation point than Phase 4's did.
MAX_TOKENS = 512


def tokenizer(base: str = BASE_MODEL):
    """The Qwen3 tokenizer, for rendering prompts only.

    Deliberately not `eval.tokenizer()`: that module imports torch at module scope, and neither the
    router nor the serving shim needs a 2.5GB tensor library to build a string. Padding side is
    irrelevant here — vLLM batches server-side, so nothing is padded client-side.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base)


def client(base_url: str = VLLM_URL, timeout: float = 300.0):
    """An OpenAI client pointed at the local vLLM server. The key is unused but must be non-empty."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=1)


@dataclass
class Completion:
    """One student answer, with everything the router and the offline sweep need about it."""

    text: str
    entities: list[dict] | None
    mean_logprob: float
    min_logprob: float
    n_tokens: int
    seconds: float = 0.0
    # A request that never reached the model. Kept distinct from `entities is None`, which means the
    # model answered and the answer was unusable. Phase 2 conflated these once and would have published
    # an 8.6% teacher schema-invalid rate against a real rate of 0% — the same trap, a different server.
    error: str | None = None

    def confidence(self, signal: str = "min") -> float:
        return self.min_logprob if signal == "min" else self.mean_logprob


def _logprobs(choice) -> list[float]:
    """The completion's per-token logprobs, defensively.

    A zero-token completion has no logprobs and no confidence to speak of; the caller turns that into
    -inf so it escalates rather than sailing past the threshold on an empty list.
    """
    lp = getattr(choice, "logprobs", None)
    values = getattr(lp, "token_logprobs", None) or []
    return [v for v in values if v is not None]


def complete(cli, model: str, prompt: str, max_tokens: int = MAX_TOKENS,
             retries: int = 1) -> Completion:
    """One greedy completion from vLLM, parsed and scored for confidence.

    `temperature=0` and the token cap match `src/pii/eval.py` exactly, so a quality number measured
    through here is comparable with every number Phases 3 and 4 produced.

    Never raises. A 1,000-row pass runs on a metered box, and one transport error propagating out of a
    thread pool would discard 999 completed requests along with the GPU-minutes that bought them. The
    failure is returned as a `Completion` carrying an `error` instead, so the caller can report it as
    what it is rather than as a model defect.
    """
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            resp = cli.completions.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
                logprobs=1,
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry once, then record rather than crash the run
            if attempt == retries:
                return Completion("", None, float("-inf"), float("-inf"), 0,
                                  round(time.perf_counter() - t0, 3),
                                  error=f"{type(exc).__name__}: {str(exc)[:160]}")
    seconds = time.perf_counter() - t0

    choice = resp.choices[0]
    lps = _logprobs(choice)
    return Completion(
        text=choice.text,
        entities=_parse(choice.text),
        # An empty completion escalates: no tokens means no evidence, not high confidence.
        mean_logprob=sum(lps) / len(lps) if lps else float("-inf"),
        min_logprob=min(lps) if lps else float("-inf"),
        n_tokens=len(lps),
        seconds=round(seconds, 3),
    )


def should_escalate(c: Completion, threshold: float, signal: str = "min") -> bool:
    """Schema failure escalates unconditionally; otherwise it is the threshold's call.

    `threshold=-inf` is the student-only arm (nothing escalates but genuine schema failures) and
    `threshold=inf` forces every request to the teacher. Both are used as smoke tests: an escalation
    path that has never executed is not a working escalation path.
    """
    return c.entities is None or c.confidence(signal) < threshold


def quantile(values: list[float], q: float) -> float:
    """The confidence threshold that escalates the lowest-confidence `q` of rows.

    The sweep is parameterised by target escalation rate rather than by raw logprob, because a rate is
    the knob an operator has ("I will pay the API for 5% of traffic") and a logprob is not. This maps
    one to the other against the observed distribution.

    The endpoints are exact rather than approximate: q=0 returns -inf so nothing clears it, and q=1
    returns +inf so everything does. Those two are the student-only and teacher-only arms, and the
    sweep is bounded by them.
    """
    if q <= 0:
        return float("-inf")
    if q >= 1:
        return float("inf")
    v = sorted(values)
    return v[min(int(q * len(v)), len(v) - 1)]


def blend(student_preds: list, teacher_preds: list, escalate: list[bool]) -> list:
    """The router's output: the teacher's answer where it was called, the student's everywhere else."""
    return [t if e else s for s, t, e in zip(student_preds, teacher_preds, escalate, strict=True)]


def route(text: str, *, student, teacher, tok, student_model: str, teacher_model: str,
          threshold: float, signal: str = "min") -> dict:
    """Student first, teacher only if the student is unsure. Returns the answer and which arm served it.

    The teacher is called with its own full ~475-token prompt through `teacher.extract`, unchanged —
    escalation is worth paying for precisely because it is the model the student was distilled from,
    measured at 0.822 on the sealed test set.
    """
    from src.pii.teacher import extract

    c = complete(student, student_model, render_prompt(tok, text, SHORT_SYSTEM))
    if not should_escalate(c, threshold, signal):
        return {"entities": c.entities, "arm": "student", "confidence": c.confidence(signal)}

    entities, usage = extract(teacher, teacher_model, text)
    return {
        "entities": entities if entities is not None else c.entities,
        "arm": "teacher",
        "confidence": c.confidence(signal),
        "escalated_because": "schema-invalid" if c.entities is None else "low-confidence",
        "teacher_tokens": usage.prompt_tokens + usage.completion_tokens,
    }

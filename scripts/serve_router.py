"""Phase 5: the served endpoint — student first, teacher on low confidence.

This is the spec's done-when made literal: a running service with a working escalation path back to
the teacher. It is a shim, deliberately — vLLM already serves the student well, and the only thing
missing is the decision about when the student's answer is not good enough.

  bash scripts/serve_vllm.sh &                       # the student, on :8000
  uvicorn scripts.serve_router:app --port 8080       # this, in front of it

  curl -s localhost:8080/extract -H 'content-type: application/json' \
       -d '{"text": "Call Dacian Ionescu on 0722 114 558."}'

  python scripts/serve_router.py --smoke              # both arms, without HTTP

The threshold comes from `benchmark_router.py`'s sweep on `val` and is passed in rather than baked in:
it is an operating point, not a constant, and the report states which one was chosen and what it cost.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.router import VLLM_URL, client, route, tokenizer  # noqa: E402
from src.pii.teacher import client as teacher_client  # noqa: E402

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "pii-student")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "accounts/fireworks/models/qwen3p7-plus")
SIGNAL = os.environ.get("ROUTER_SIGNAL", "min")
# -inf routes everything to the student except genuine schema failures — the safe default for a
# service started before the sweep has picked an operating point. Set it explicitly in production.
THRESHOLD = float(os.environ.get("ROUTER_THRESHOLD", "-inf"))

# Built once at import: the tokenizer is ~1s to load and the clients hold connection pools. Doing this
# per request would put model loading inside the latency this project spent Phase 4 measuring.
_tok = None
_student = None
_teacher = None


def _deps():
    global _tok, _student, _teacher
    if _tok is None:
        _tok = tokenizer()
        _student = client(os.environ.get("VLLM_URL", VLLM_URL))
        _teacher = teacher_client()
    return _tok, _student, _teacher


def extract(text: str, threshold: float = THRESHOLD) -> dict:
    tok, student, teacher = _deps()
    return route(text, student=student, teacher=teacher, tok=tok, student_model=STUDENT_MODEL,
                 teacher_model=TEACHER_MODEL, threshold=threshold, signal=SIGNAL)


def build_app():
    """FastAPI is imported lazily so --smoke and the test suite do not require it."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Request(BaseModel):
        text: str
        # Per-request override, so the escalation path can be exercised against a live service
        # without restarting it. An escalation path that has never executed is not a working one.
        threshold: float | None = None

    api = FastAPI(title="PII router")

    @api.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "student": STUDENT_MODEL, "threshold": THRESHOLD, "signal": SIGNAL}

    @api.post("/extract")
    def extract_route(req: Request) -> dict:
        return extract(req.text, THRESHOLD if req.threshold is None else req.threshold)

    return api


app = build_app() if os.environ.get("ROUTER_EAGER_APP", "1") == "1" else None

SMOKE_TEXT = "Email Dacian Cosmin Ionescu at d.ionescu@example.ro or call 0722 114 558."


def smoke() -> None:
    """Both arms, in one command. The student answers, then everything is forced to the teacher."""
    for label, threshold in (("student arm", float("-inf")), ("teacher arm", float("inf"))):
        out = extract(SMOKE_TEXT, threshold)
        n = len(out["entities"] or [])
        print(f"{label:<12} arm={out['arm']:<8} entities={n:<3} conf={out['confidence']:.3f}"
              + (f"  because={out.get('escalated_because')}" if out["arm"] == "teacher" else ""))
        assert out["arm"] == ("teacher" if threshold == float("inf") else "student"), \
            f"{label} routed to {out['arm']}"
        assert out["entities"], f"{label} returned nothing"
    print("both arms fire")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="exercise both arms and exit")
    if ap.parse_args().smoke:
        smoke()
    else:
        print(__doc__)

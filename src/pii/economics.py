"""The arithmetic the break-even volume is made of.

Phase 4 computed cost per 1,000 requests inline in `scripts/write_phase4.py`; Phase 5 needs the same
amortisation plus a break-even volume and a router premium, in three callers (the sweep, the report and
the test). It lives here so both phases divide by the same denominator — a Phase 5 that re-derived
`per_1k` would produce a before/after difference that measures the arithmetic.

Nothing here reads a file or calls anything. Every function is a pure expression over measured inputs,
which is what makes `tests/test_phase5_math.py` able to pin the verdict the report turns on.
"""

SECONDS_PER_DAY = 86_400
HOURS_PER_DAY = 24


def per_1k(rate_usd_h: float, requests_per_s: float, utilisation: float = 1.0) -> float:
    """GPU-hour cost amortised over measured throughput, with idle time charged for.

    Lifted unchanged from Phase 4 (D-029): a card billed by the hour costs the same whether or not
    requests arrive, so `utilisation` below 1.0 is not a discount — it is the same hourly bill spread
    over fewer requests, and the cost per request rises.
    """
    denom = requests_per_s * 3600 * utilisation
    return float("inf") if denom <= 0 else rate_usd_h / denom * 1000


def capacity_per_day(requests_per_s: float) -> float:
    """The most one card can serve in a day. A break-even volume above this is unreachable on it."""
    return requests_per_s * SECONDS_PER_DAY


def break_even_per_day(rate_usd_h: float, teacher_per_request: float,
                       escalation_rate: float = 0.0) -> float:
    """Requests per day above which self-hosting beats paying the API. Below it, rent.

    At V requests/day the two arms cost:

        self-hosted = 24·C + V·e·p_t     one card, plus the API bill the escalated fraction still incurs
        API-only    = V·p_t

    Setting them equal gives V* = 24·C / (p_t·(1 − e)). The router therefore *raises* the break-even
    volume: every escalated request is one the GPU was paid for and the API is billed for anyway, so the
    per-request saving shrinks by exactly the escalation rate. That premium is a real cost of the recall
    escalation buys, and reporting the break-even without it would hide the router's price.

    Returns inf when nothing can amortise the card — a rate of zero requests saved per request served.
    """
    saving_per_request = teacher_per_request * (1 - escalation_rate)
    if saving_per_request <= 0:
        return float("inf")
    return HOURS_PER_DAY * rate_usd_h / saving_per_request


def capacity_fraction(rate_usd_h: float, teacher_per_request: float, requests_per_s: float,
                      escalation_rate: float = 0.0) -> float:
    """Where the break-even volume sits as a fraction of what one card can serve in a day.

    This is the number that decides whether a break-even volume is *usable*, and it is reported rather
    than collapsed to a yes/no because the margin is the whole story. Phase 4 came out at 0.996: the
    student was $0.4349 per 1,000 against the API's $0.4366, so it technically broke even — but only on
    a card saturated every second of every day, with no headroom for a burst and none for the idle time
    real traffic leaves behind. That is what "no break-even volume" meant in D-032, and a strict
    inequality would have called it a win.

    Below ~1 the card starts paying before it runs out of capacity, and the lower the number the more
    headroom there is. At or above 1 there is no volume the card can both reach and profit at.

    Adding cards does not move it: cost and capacity scale together, so the ratio is invariant.
    """
    capacity = capacity_per_day(requests_per_s)
    if capacity <= 0:
        return float("inf")
    return break_even_per_day(rate_usd_h, teacher_per_request, escalation_rate) / capacity


def blended_per_1k(rate_usd_h: float, requests_per_s: float, teacher_per_request: float,
                   escalation_rate: float = 0.0, utilisation: float = 1.0) -> float:
    """Cost per 1,000 requests through the router: the GPU, plus the API for what it escalates.

    The GPU term is charged on all 1,000 — an escalated request still ran on the student first, and that
    is the honest accounting. Escalation is not a discount on the card.
    """
    return per_1k(rate_usd_h, requests_per_s, utilisation) + 1000 * escalation_rate * teacher_per_request

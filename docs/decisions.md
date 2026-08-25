# Decision log

Append-only. Written as decisions are made, not reconstructed afterwards. Feeds the tradeoffs section of the
final README.

---

## D-001 · Task = PII detection on English text

**Date:** 2026-08-25
**Chose:** PII entity detection over free English text.
**Over:** SQL generation against one schema, multi-label classification, query rewriting.
**Why:** Three properties the other candidates lack together. It is automatically scorable against gold
labels (no LLM judge needed, so no judge cost and no judge bias). Its output is schema-validatable, which
gives Phase 5's router a free escalation trigger. And "be conservative, escalate when unsure" is the
genuinely correct business behaviour for redaction, so the router is not a benchmark contrivance.
**Revisit if:** the teacher ceiling comes in low enough that the task is not actually one a large model
already does well — that would violate the premise of distillation (transferring a capability, not creating
one).

---

## D-002 · Dataset = `ai4privacy/pii-masking-openpii-1.5m`

**Date:** 2026-08-25
**Chose:** `ai4privacy/pii-masking-openpii-1.5m` (~1.64M rows, 30 languages, 19 PII classes, CC-BY-4.0).
**Over:** `ai4privacy/open-pii-masking-500k-ai4privacy`, `pii-masking-200k`.
**Why:** Licensing, decided on a direct read of the dataset cards. The 500k release was generated using
Llama 3.1/3.3 models, so its README places it under the Llama Community License: *"If you use this dataset to
create, train, fine-tune, or improve an AI model that you distribute, you must include 'Llama' at the
beginning of the model name"*, plus a requirement to display *"Built with Llama"*. Complying would mean
shipping a Qwen-derived model named `Llama-...`, which is both confusing and directly contradicts the
Qwen-teaches-Qwen design in D-005. The 1.5M release is CC-BY-4.0 with no Llama derivation and no naming
clause. `pii-masking-200k` does not state its license clearly on the card, which is its own risk for a public
repo.
**Revisit if:** never, realistically — but if the 1.5M English subset turns out too small or too synthetic,
the fallback is `pii-masking-openpii-1m` (also CC-BY-4.0, same lineage), not the 500k.

---

## D-003 · Output format = `{label, value}` list, no character offsets

**Date:** 2026-08-25
**Chose:** Model returns `{"entities": [{"label": ..., "value": ...}]}`.
**Over:** character-offset spans `{start, end, label}` matching `privacy_mask` natively; full masked-text
rewrite matching `masked_text`.
**Why:** Offsets would measure the wrong thing. LLMs are unreliable at counting character positions, so
off-by-N errors would dominate the error budget and the benchmark would report arithmetic ability rather than
PII detection ability — contaminating the one number the whole project rests on. Offsets are recoverable
post-hoc with `str.find` when they are actually needed. The masked-text rewrite was rejected on cost and
attribution: it costs roughly 4x the output tokens, and a single dropped word in the copied carrier text
scores as a detection failure.
**Revisit if:** a downstream consumer needs true offsets for overlapping entities, where `str.find` recovery
is ambiguous.

---

## D-004 · Teacher = open-weight model on Fireworks, not Claude

**Date:** 2026-08-25
**Chose:** A large open-weight Qwen model served on Fireworks (candidate set decided by bake-off, D-007).
**Over:** Claude Sonnet 5 via the Anthropic API — the standard frontier-teacher recipe (Alpaca, Vicuna, Orca).
**Why:** Budget, stated honestly. Available funding is $50 of Fireworks credits and no Anthropic credits, and
Fireworks does not host Claude. Fireworks prices every model above 16B params at a flat $0.90/M tokens with a
50% batch discount, which puts the full 5-10k-example Phase 2 generation run comfortably inside budget.
**Tradeoff accepted:** a frontier teacher would likely set a higher ceiling on the hard cases — ambiguous or
non-Western names, mixed-script text, and PII that depends on surrounding context rather than surface form.
The reported ceiling is therefore a ceiling for *this* teacher, not for the task.
**Side effect, in our favour:** with an open teacher and an open student, "rent vs own" compares two things
that could both actually be self-hosted, and pairing it with D-005 isolates scale as the only variable.
**Revisit if:** API credits become available — re-running Phase 1's bake-off with a frontier teacher would
raise the ceiling and is cheap (~$1).

---

## D-009 · Teacher candidates re-picked from the live Fireworks catalogue

**Date:** 2026-08-25
**Chose:** Bake off six candidates spanning a 20x price range — `qwen3p7-plus`, `deepseek-v4-pro`, `glm-5p2`,
`gpt-oss-120b`, `minimax-m3`, `qwen3p8-2p4t-a95b`.
**Over:** the planned `Qwen2.5-72B-Instruct` vs `Qwen3-235B` comparison.
**Why:** `Qwen2.5-72B` is **no longer served on Fireworks serverless** — nor is any Llama model or
`Qwen3-235B`. This was caught by querying the live catalogue instead of hardcoding the model IDs the plan
assumed, which is the reason `scripts/list_models.py` exists. Since the intended teacher no longer exists, the
candidate set was rebuilt to span the price range rather than to find a single "best" model: the deliverable
is an F1-per-dollar curve, which is the same unit-economics question the whole project asks.
**Knock-on:** this invalidates the same-family design in D-005 — see D-011.
**Revisit if:** the catalogue changes again. It already did once, so re-run `list_models.py` before Phase 2.

---

## D-010 · Pricing table corrected — the flat-rate assumption was wrong

**Date:** 2026-08-25
**Chose:** A per-model pricing table read from Fireworks' docs, with the flat >16B rate only as a fallback.
**Over:** the earlier assumption that every model above 16B params bills at a flat $0.90/M both ways.
**Why:** The flat rate applies only to models Fireworks does not price individually. The named models vary by
more than 20x: `gpt-oss-120b` at $0.15/$0.60 against `kimi-k3` at $3.00/$15.00, with the intended workhorse
`qwen3p7-plus` at $0.40/$1.60 — cheaper than the assumed flat rate, while `deepseek-v4-pro` ($1.74/$3.48) is
nearly 4x it on output. Cost estimates built on the flat assumption would have been wrong in both directions,
and cost accuracy is the point of this project.
**Revisit if:** before every spending run — rates move, and the table records the date it was checked.

---

## D-005 · Student = Qwen2.5-7B-Instruct, same family as the teacher

**Date:** 2026-08-25
**Chose:** `Qwen2.5-7B-Instruct` fine-tuned with QLoRA.
**Over:** `Llama-3.1-8B-Instruct` (more battle-tested fine-tuning ecosystem).
**Why:** Keeping teacher and student in the same model family means the distillation measures the effect of
*scale* rather than confounding scale with vendor, tokenizer, and pretraining-data differences. That is a
cleaner experimental claim and mirrors how distillation is done in the literature. Qwen2.5's tokenizer also
handles multilingual text better, which keeps the door open to the dataset's other 29 languages as a stretch
goal.
**Revisit if:** Qwen2.5-7B proves unstable under QLoRA on the available hardware; Llama-3.1-8B is the fallback
and the family-symmetry argument is then dropped from the writeup rather than quietly kept.
**Status: superseded in part — see D-011.**

---

## D-011 · Same-family design deferred pending bake-off results

**Date:** 2026-08-25
**Chose:** Defer the student decision until the teacher bake-off reports.
**Why:** D-005's argument depended on a Qwen2.5 teacher, and Qwen2.5-72B is gone from the catalogue (D-009).
Three outcomes are possible and only the measurement distinguishes them:
1. `qwen3p7-plus` is competitive → switch the student to **Qwen3-8B**, restoring family symmetry cleanly at
   the same generation as the teacher.
2. A non-Qwen teacher wins decisively → take the higher ceiling and **state plainly that teacher and student
   are unrelated families**, dropping the "isolates scale" claim rather than quietly keeping it.
3. The field is close → prefer the Qwen teacher as a tiebreak and keep the cleaner experimental story.
Deferring costs nothing: the student only matters from Phase 3, and the bake-off runs first regardless.
**Revisit if:** resolved by the bake-off — this entry gets a decisive outcome appended.

**Resolved (student half):** the student is **Qwen3-8B**. It matches the generation of `qwen3p7-plus`, so if
that model wins the teacher comparison the same-family design of D-005 is restored cleanly; QLoRA on a 16GB
T4 is confirmed feasible. `Qwen2.5-7B` was dropped — it is now older than every available teacher, so it
gained nothing over Qwen3-8B. `Qwen3-4B` was considered and rejected for now: it would improve serving cost
and halve training time on the Kaggle quota, but the spec's headline claim is an 8B student and a weaker
student risks understating the quality number. It stays the fallback if 8B proves impractical on a T4.

---

## D-012 · Teacher price is a framing choice, not just a cost

**Date:** 2026-08-25
**Chose:** Pick the teacher to model a realistic scenario, and publish a break-even **sensitivity table**
across price tiers rather than only a single number.
**Why:** The break-even volume is the point where GPU cost overtakes API cost, so it is directly proportional
to the teacher's per-token price. On the same student and GPU, an illustrative sweep spans roughly 2,400
req/day against `kimi-k3` ($3.00/$15.00) to ~54,000 req/day against `gpt-oss-120b` ($0.15/$0.60) — a 20x
swing driven entirely by which teacher is named. Choosing the cheapest teacher would quietly rig the project
against its own thesis; choosing the priciest would rig it in favour. Reporting the sweep alongside the
headline makes the analysis honest and shows it generalises beyond one vendor's price list.
**Revisit if:** never — but the table must be rebuilt from *measured* throughput and GPU cost in Phase 5, not
from these planning estimates.

---

## D-006 · Label scope = ~12 of the 19 classes, chosen by measured frequency

**Date:** 2026-08-25
**Chose:** The top ~12 PII classes by frequency in the English subset, selected empirically after counting.
**Over:** all 19 classes; a ~15-class middle ground.
**Why:** The spec calls for one *narrow* task. Long-tail classes would have too few examples at 5-10k rows to
learn or to score reliably, so their noisy per-label F1 would drag the headline number down for reasons that
have nothing to do with distillation quality. Selecting by measured frequency rather than by intuition avoids
baking in a guess about which classes matter.
**Measured, then chosen.** Label frequency over a 60k-row sample of the English subset showed a sharp cliff:
20 classes appear regularly (TIME, the rarest, at 745), then a tail of 9 classes with fewer than 10
occurrences each (`URL`, `SALARY`, `ACCOUNTNUM`, …) that leak in from other source datasets. Within the
learnable 20, a strict top-12-by-frequency cut would have kept `TITLE` ("Mr", "Master", 3766) and `AGE`
(2615) while dropping `CREDITCARDNUMBER` (1868), `TAXNUM` (1515) and `SOCIALNUM` (1176) — i.e. a redactor
that masks honorifics but leaves card numbers and national insurance numbers in the clear. Frequency was
therefore used as a *floor* (everything kept has >1,100 occurrences, so all 12 are learnable) and redaction
value as the *selector*.
**Final set:** GIVENNAME, SURNAME, DATE, EMAIL, CITY, TELEPHONENUM, STREET, ZIPCODE, IDCARDNUM,
CREDITCARDNUMBER, TAXNUM, SOCIALNUM.
**Revisit if:** the student saturates the 12-class task, in which case widening to all 20 is a natural
stretch goal.

---

## D-007 · Reservoir-sample the splits instead of taking the first N rows

**Date:** 2026-08-25
**Chose:** A single full pass over each source split with reservoir sampling.
**Over:** Pooling the first ~24k eligible rows and shuffling (the original approach), which is much faster.
**Why:** A bug caught by inspecting the manifest rather than trusting it. The source `train.jsonl` is ordered
by `source_dataset`, and the **first ~150,000 rows are 100% Singapore-region**; CA/GB/US/IN only begin
appearing after that, evenly balanced from there on. The first build therefore produced splits that were
**83% SG**. The splits were still internally consistent — train, val and test were all skewed the same way,
so the comparison would have been valid — but the task would silently have been "PII in Singapore-formatted
documents", and any claim about generalisation would have been wrong. Reservoir sampling gives a uniform draw
over the entire split at O(n) memory; the cost is one full pass (~10 min, paid once).
**Measured outcome:** SG fell from 6,666/8,000 (83%) to 957/8,000 (12%), with GB/CA/IN/US each landing at
21-23% — 8,000 sampled from 161,426 eligible English train rows, and 2,000 from 40,340 eligible validation
rows. `data/manifest.json` records the SHA256, seed, and per-split label and region distribution.
**Revisit if:** never. The cost is a one-time build.

---

## D-008 · Keep `SEX` and `GENDER` as distinct classes (moot for now)

**Date:** 2026-08-25
**Chose:** Treat them as genuinely different classes — though both fall outside the 12 selected in D-006.
**Why:** They looked like near-duplicates on the dataset card, but inspecting real rows settled it: a single
record carries `Sex: M` alongside `Gender Identity: Two-spirit`, so they encode different attributes and
merging them would have destroyed a real distinction.
**Revisit if:** the label set is widened to all 20 classes, at which point this becomes live again and the
student's confusion matrix between the two is worth checking specifically.

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
**Verified, not assumed:** the legacy IDs were tested directly, not just checked against `models.list()`.
`qwen2p5-72b`, `qwen2p5-72b-instruct`, `qwen2p5-7b-instruct` and `llama-v3p1-70b-instruct` all return
`404 — "Model not found, inaccessible, and/or not deployed"`. Third-party pages still quoting $0.90/M for
Qwen2.5-72B on Fireworks are stale. The models exist only as **on-demand dedicated deployments** at ~$7/hr
per H100 (a 72B needs several), which was rejected on two grounds: it would consume a large share of the $50
budget, and it would replace per-token accounting with GPU-hour accounting on the *rent* side of a
rent-vs-own comparison, muddying the break-even analysis the project exists to produce.
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

**Resolved (student half):** the student is **Qwen3-8B**; QLoRA on a 16GB T4 is confirmed feasible.

**Correction (same day):** an earlier version of this entry claimed Qwen3-8B "matches the generation of
`qwen3p7-plus`" and so restored the same-family design. That was wrong. **Qwen3.7 was never released as open
weights — the generation was skipped**, so the teacher has no open sibling at any size and a true
same-generation pair is impossible, not merely inconvenient. Qwen3-8B is the original Qwen3 release, several
generations behind the teacher. The nearest open models are `Qwen3.5-9B` and `Qwen3.5-4B`, still a generation
short.
**Consequence:** the "isolates scale" argument from D-005 is **abandoned, not quietly preserved**. Teacher and
student are related in lineage but not in generation, and the writeup must say so. The student was therefore
chosen on practical grounds — documented QLoRA path on a T4, VRAM headroom, and the spec's 8B framing —
rather than on family aesthetics. `Qwen3.5-4B` remains the fallback if 8B proves impractical, and would
additionally improve the break-even number. `Qwen2.5-7B` was dropped — it is now older than every available teacher, so it
gained nothing over Qwen3-8B. `Qwen3-4B` was considered and rejected for now: it would improve serving cost
and halve training time on the Kaggle quota, but the spec's headline claim is an 8B student and a weaker
student risks understating the quality number. It stays the fallback if 8B proves impractical on a T4.

---

## D-013 · Reasoning is switched OFF for the teacher

**Date:** 2026-08-25
**Chose:** `reasoning_effort: "none"` on every teacher call, with a fallback that strips the parameter for
models that reject it.
**Over:** leaving reasoning at its default, or lowering it to `"low"`.
**Why:** The first bake-off returned micro-F1 0.548 with recall 0.436, which looked like a weak teacher. It
was not. `qwen3p7-plus` is a reasoning model: a single probe showed 389 input tokens producing **2,459 output
tokens, of which ~1,900 were thinking** and only ~200 the actual answer. That overran the 2,048 `max_tokens`
cap, truncating the JSON mid-answer, so **33.5% of outputs were unparseable** and every one of those scored
as a total miss. The metric was measuring truncation, not capability. Setting `reasoning_effort: "none"` cuts
output to ~145 tokens — a **17x reduction** — and removes the truncation entirely. `"low"` makes it *worse*
(2,896 tokens), so it is not a middle ground.
**Wider finding, and the one worth putting in the README:** thinking tokens are billed as output, so a
reasoning model is a poor economic fit for bounded extraction — a nominally cheap $0.40/$1.60 model
effectively costs ~9x its sticker rate. This cuts against the API teacher and *for* a non-reasoning
fine-tuned student, which strengthens the break-even case on measured evidence rather than assertion.
**Revisit if:** a task in a later phase genuinely benefits from deliberation. PII extraction does not — it is
a bounded lookup.

---

## D-014 · Three annotation conventions encoded in the teacher prompt

**Date:** 2026-08-25
**Chose:** State the dataset's span conventions explicitly in the system prompt.
**Over:** accepting the low per-label scores as the teacher's real ceiling.
**Why:** After the truncation fix, three labels still scored oddly — `STREET` worst at 0.129 F1. Diffing
predictions against gold (rather than trusting the aggregate) showed all three were **annotation-convention
mismatches, not detection failures**:
- `STREET`: gold excludes the building number — `"1222 Chanditala Road"` is `BUILDINGNUM` + `"Chanditala
  Road"`. Since `BUILDINGNUM` is out of scope (D-006) the model had nowhere to put the number and folded it
  into the street.
- `GIVENNAME`: gold is a **single span for all given names** (`"Dacian Cosmin"`), while the model emitted one
  entity per name.
- `DATE`: gold keeps the full timestamp (`"2016-11-25T00:00:00"`); the model normalised it to `"2016-11-25"`.
All three now verified fixed on sample examples.
**Why this matters beyond the fix:** the headline F1 was hiding three distinct bugs, none of them about the
model's ability to *find* PII. Any teacher ceiling measured before this would have been understated, and the
student would then have been flattered by comparison against it.
**Revisit if:** the label set changes — conventions are dataset-specific and would need re-deriving.

---

## D-015 · IDCARDNUM scope leak: fixed the precision, lost the recall — kept anyway

**Date:** 2026-08-25
**Chose:** Keep the tightened prompt (micro-F1 0.832) and stop iterating, despite it making `IDCARDNUM`
worse.
**Over:** reverting to the looser prompt (0.828), or iterating further on the label.
**Why:** Confusion analysis showed `IDCARDNUM` precision was 0.42 on *both* candidate teachers — an identical
figure across two vendors, which is a strong signal of a systematic scope problem rather than model weakness.
The cause was the same class of bug as `STREET`/`BUILDINGNUM`: `DRIVERLICENSENUM` and `PASSPORTNUM` are out
of scope (D-006), so the model had nowhere to put them and folded them into `IDCARDNUM`
(`"Driver Licence No.: KRITS.910081.KL.629"` → `IDCARDNUM`, gold: nothing).

Instructing the model to reserve `IDCARDNUM` for national ID cards only did work — precision rose 0.423 →
0.812 — but it **over-corrected**: recall collapsed 0.887 → 0.245, because the model can no longer tell a
national ID number from a licence number and now omits both. Per-label F1 got *worse*: 0.573 → 0.377.

Kept regardless, on three grounds: overall micro-F1 was marginally better (0.828 → 0.832), precision improved
across the board (0.806 → 0.847), and hallucinated values fell to zero. **For distillation, label precision
matters more than recall** — a wrong label actively teaches the student an error, while a missing one merely
teaches less. The honest read is that 0.828 vs 0.832 is noise at n=200; the real gain is cleaner training
labels.
**Cost of the lesson:** one iteration, $0.09.
**Revisit if:** Phase 4 shows the student inheriting the under-detection. The genuine fix is not prompting but
**adding `DRIVERLICENSENUM` and `PASSPORTNUM` back to the label set** so the model has somewhere legitimate to
put them — deferred because it would re-open D-006 and re-freeze the splits.

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

## D-016 · Phase 2 generation runs synchronously, declining the 50% batch discount

**Date:** 2026-08-25
**Chose:** Reuse the existing synchronous `extract()` path with a 16-thread pool for all 8,000 train rows.
**Over:** Building a Fireworks batch-inference path, which halves the per-token rate.
**Why:** The batch API is quicker on neither axis that matters. Wall clock: a thread pool finishes in
well under an hour and is watchable; a batch job is queue-dependent and unattended for an unknown number of
hours. Build time: the synchronous path is already verified against 200 live calls, while the batch flow is a
separate API surface that would need writing and debugging before it could be trusted with the run. What it
buys is **$1.89** — the difference between $3.77 and $1.88 — against a $50 budget with ~$1.50 spent. Paying
$1.89 to keep an expensive run interactive and resumable is the right trade at this scale.
**Reported, not hidden:** `reports/phase2.md` prints the batch alternative next to the actual spend, so the
declined saving is visible rather than quietly omitted from a project about cost transparency.
**Revisit if:** a later phase needs a generation run an order of magnitude larger, where the discount stops
being rounding error and the unattended latency stops mattering.

---

## D-017 · The student is trained on a ~60-token prompt, not the teacher's ~475-token one

**Date:** 2026-08-25
**Chose:** `student.SHORT_SYSTEM` — one instruction line, the 12 labels, and the output shape.
**Over:** Training the student on `teacher.SYSTEM` verbatim, for a maximally apples-to-apples comparison.
**Why:** The teacher's prompt is long because it has to be: ~475 of its 628 input tokens per request are the
annotation conventions from D-014, spelled out because a prompted model has no other way to learn them. A
fine-tuned student learns them from the weights instead, so at serving time it only needs to be told what
shape to emit. That cuts input tokens per request roughly **4x**, and since the break-even volume is set by
the student's cost per request against the teacher's, this feeds the project's headline number directly. It
is also the honest form of the "own beats rent" argument: not needing a long prompt is a *real* advantage of
owning the model, not an accounting trick.
**Tradeoff accepted:** teacher and student no longer see identical prompts. This does not contaminate the
comparison — both are scored on output against the same gold, and the metric never sees the prompt — but the
writeup must state it rather than implying a matched setup.
**Consequence:** train-time and serve-time prompts must be byte-identical or the student silently degrades,
which is why `SHORT_SYSTEM` lives in one importable module used by Phase 3 and Phase 5 alike, with a test
asserting the exact bytes of the assistant turn.
**Revisit if:** Phase 4 shows the student failing a specific convention (given-name grouping, street numbers,
verbatim dates). The fix would be adding that one rule back to the short prompt, not restoring all 475 tokens.

---

## D-018 · Teacher outputs are repaired, not filtered against gold

**Date:** 2026-08-25
**Chose:** Drop schema-invalid outputs, strip entities whose value does not occur in the source text, and
keep everything else — including examples the teacher simply got wrong.
**Over:** Also dropping examples whose per-example F1 against train gold falls below a threshold.
**Why:** The train split has gold labels, so a gold-agreement filter is *available* — but using it would
quietly change what the project claims. The thesis is distillation: the student learns from the teacher, and
a real deployment generating training data from a frontier model has no gold to filter against. Filtering
would also train the student only on the examples the teacher found easy, which risks a student that looks
good on the benchmark and fails on exactly the hard cases Phase 5's router is supposed to escalate.
The two repairs that *are* applied need no gold and are unarguable: unparseable output contains nothing to
learn, and a value absent from the input can never be a correct extraction — teaching it would teach the
student to invent, which is the worst failure mode for a redactor.
**Measured, not applied:** `reports/phase2.md` reports how many rows survive and what label quality results at
thresholds 0.0/0.5/0.7/0.9/1.0, so the option is costed and available to Phase 3 without being taken now.
**Revisit if:** Phase 3's student underperforms and the loss curves suggest label noise rather than capacity.
The sweep says what the trade costs in training rows before any of it is re-run.

---

## D-008 · Keep `SEX` and `GENDER` as distinct classes (moot for now)

**Date:** 2026-08-25
**Chose:** Treat them as genuinely different classes — though both fall outside the 12 selected in D-006.
**Why:** They looked like near-duplicates on the dataset card, but inspecting real rows settled it: a single
record carries `Sex: M` alongside `Gender Identity: Two-spirit`, so they encode different attributes and
merging them would have destroyed a real distinction.
**Revisit if:** the label set is widened to all 20 classes, at which point this becomes live again and the
student's confusion matrix between the two is worth checking specifically.

---

## D-019 · fp16 and SDPA, because the T4 is sm_75

**Date:** 2026-08-25
**Chose:** `fp16=True`, gradient checkpointing (non-reentrant), SDPA attention.
**Over:** the bf16 + FlashAttention-2 recipe every current QLoRA tutorial ships.
**Why:** Kaggle's T4s are compute capability 7.5. bf16 needs Ampere (8.0+) and FlashAttention-2 needs Ampere
too; both fail at *model load*, before a single step, so a copied recipe does not degrade — it simply does
not run. Non-reentrant checkpointing is the third piece: the reentrant path does not see LoRA parameters as
requiring grad and dies with "none of the inputs have requires\_grad". The script asserts the compute
capability up front and prints which path it took, so this is a sentence rather than a traceback.
**Cost:** fp16 needs loss scaling and is slightly less numerically forgiving than bf16, and the lack of FA2
costs attention throughput at these sequence lengths. Both are accepted in exchange for a $0 training bill.
**Revisit if:** training moves to a rented Ampere GPU, which would also settle the Phase 5 serving question.

---

## D-020 · `lora_alpha = 2 × rank`, so rank is the only variable

**Date:** 2026-08-25
**Chose:** alpha 16 at rank 8, alpha 64 at rank 32.
**Over:** the common default of a fixed `lora_alpha=16` across both configurations.
**Why:** PEFT scales the adapter update by `alpha / r`. Holding alpha fixed while rank moves from 8 to 32
therefore cuts the effective update size by 4x at the same time as it quadruples capacity, and the two
effects land in opposite directions. A result from that comparison could not distinguish "rank 32 has more
capacity" from "rank 32 was effectively trained at a quarter of the learning rate" — which would make the
spec's requested trade-off unreadable. Tying alpha to rank holds the scaling constant at 2.0 and leaves
adapter capacity as the single difference.
**Revisit if:** a rank sweep is ever run for absolute quality rather than for a two-point comparison, where
tuning alpha per rank is legitimate.

---

## D-021 · Per-epoch eval on 200 rows; the full 1,000 only on the final pick

**Date:** 2026-08-25
**Chose:** score the first 200 val rows after each epoch, and all 1,000 once on the selected adapter.
**Over:** the full 1,000-row val set at every epoch.
**Why:** Generation, not training, is the expensive half of an eval on a 4-bit 8B on a T4. Six full evals
across two configurations would cost more GPU-hours than the training runs they are meant to monitor, inside
a ~30h weekly quota. 200 is also the exact n the teacher ceiling was measured at, so the student's per-epoch
curve and the teacher's 0.832 sit on one axis without a sample-size caveat between them.
**Also chose:** select the best checkpoint by **val micro-F1, not val loss**. They disagree, and F1 is the
number the project reports; picking on loss would optimise a proxy.
**Cost:** ±~0.03 F1 of sampling noise on the per-epoch curve. Acceptable for choosing an epoch, which is why
the headline number is re-measured on all 1,000.
**Revisit if:** the two configurations land within noise of each other, in which case the tie is broken on
the full val set rather than on 200 rows.

---

## D-022 · `enable_thinking=False`, frozen in one function and asserted by test

**Date:** 2026-08-25
**Chose:** all prompt rendering goes through `student.render_prompt`, which passes `enable_thinking=False`.
**Over:** calling `apply_chat_template` at each site, which is what every example does.
**Why:** Qwen3 is a hybrid reasoning model and its template is inconsistent in a way that defeats the obvious
code. Measured directly:

| call | think block |
|---|---|
| `add_generation_prompt=True`, flag unset | **absent** |
| `add_generation_prompt=True, enable_thinking=False` | **present** — `<think>\n\n</think>\n\n` |
| a full conversation with an assistant turn, any flag | **always present** |

Training renders the third form, so an empty think block precedes every training answer. Serving via the
default renders the first, and the student is handed a prefix it has never seen. Nothing raises; the model
just scores lower, and the evidence points at the fine-tune rather than at the plumbing. This is the same
class of failure as D-013 — a generation-time flag on a reasoning model silently invalidating a measurement —
found before it cost anything this time rather than after.
**Consequence:** `render_prompt` is the only renderer, and `tests/test_prompt_render.py` asserts against the
real tokenizer that prompt + completion reconstructs the training render exactly. `transformers` and `jinja2`
were added to `requirements.txt` (tokenizer only, no torch) specifically so that check runs locally in
seconds instead of five hours into a Kaggle session.
**Revisit if:** the student is ever swapped for a non-Qwen base, where the template's behaviour must be
re-measured rather than assumed to match.

---

## D-023 · Plain HF `Trainer` with explicit label masking, not TRL

**Date:** 2026-08-25
**Chose:** `transformers.Trainer`, tokenising prompt and completion separately and setting `labels = -100`
across the prompt.
**Over:** `trl.SFTTrainer`, the standard tool for exactly this job.
**Why:** The one feature TRL is wanted for here does not work on this model. `assistant_only_loss=True`
requires the chat template to declare `{% generation %}` markers; Qwen3's template has none — checked, not
assumed — so it raises rather than falling back. The older `DataCollatorForCompletionOnlyLM` matches on a
response-template *string* and has churned repeatedly across TRL releases. The explicit version is about ten
lines, and separating the halves is provably safe: over 500 rows, tokenising prompt and completion
separately gives token-for-token the same ids as tokenising them joined.
**Also buys:** the masking is inspectable. Measured on real rows, only 33% of tokens are answer, so the mask
is not a detail — training on the full sequence would spend two thirds of the gradient teaching the model to
reproduce a system prompt it is given for free at inference.
**Revisit if:** Qwen ships a template with generation markers, or the training loop needs something TRL
provides and this does not (packing, DPO, or a reward model).

---

## D-024 · Two epochs, chosen on measured throughput

**Date:** 2026-08-26
**Chose:** 2 epochs per configuration, per-device batch 4 x grad-accum 4 (effective 16).
**Over:** the 3 epochs the Phase 3 plan assumed.
**Why:** Measured on the target hardware rather than estimated. At **24.44 s/step** a 3-epoch run over
7,842 examples is **10.5h**, past Kaggle's ~9h session cap, so it could only complete by resuming across
two sessions. 2 epochs is **7.0h** and finishes unattended in one, which takes the least-tested code in the
phase — the resume path — off the critical path for both configurations.

The epoch count is not a stopping decision that can be deferred: `lr_scheduler_type="cosine"` decays over
the declared total, so a 3-epoch schedule halted at epoch 2 leaves the model mid-decay at a still-high LR
and is strictly worse than a 2-epoch schedule that completes. The number has to be committed up front.

Supporting evidence that epoch 3 was unlikely to pay: a single epoch over a **256-example** subset (16
optimizer steps) already reached **0.692 micro-F1** against the teacher's 0.832, at 0.5% schema-invalid,
with `EMAIL` 0.989 and `TELEPHONENUM` 0.898. Training loss was 0.158. The capability transfers fast, and
the spec's own guidance is to watch for overfitting after two or three epochs, not to assume three helps.
**Cost, stated plainly:** if the epoch-1 to epoch-2 curve is still climbing, this leaves quality on the
table and the report must say so rather than presenting 2 as optimal. The per-epoch val curve is what
shows whether that happened.
**Revisit if:** the two-epoch curves are still rising at the end, in which case a 3-epoch run is worth
21h of quota across two sessions — or training moves to a GPU where 3 epochs fits a single session.

---

## D-025 · The frozen embedding and output head stay in fp16

**Date:** 2026-08-26
**Chose:** recast non-trainable fp32 tensors over 100M parameters back to fp16 after
`prepare_model_for_kbit_training`, and run generation under `torch.autocast`.
**Over:** accepting PEFT's blanket fp32 upcast.
**Why:** The upcast exists so fp16 training stays numerically stable, and for layer norms that is right and
nearly free. For Qwen3's **151,936-token vocabulary** it is neither: `embed_tokens` and `lm_head` are ~622M
parameters each, and upcasting them cost **2.32GiB of a 14.56GiB card** — measured, resident went 5.66 to
7.98 GiB. Neither tensor is trained here (LoRA targets only the attention and MLP projections), so neither
carries an optimiser state or receives a gradient, and the precision buys nothing.
**Effect:** per-device batch 4 became affordable where it had OOMed, taking training from 39.79 to 24.44
s/step — **1.63x**, and the difference between a run that fits a Kaggle session and one that does not.
**Consequence:** the recast exposed a latent dtype mismatch. Layer norms remain fp32 by design, so the final
norm emits an fp32 activation into an fp16 output projection. Training never hit it because AMP inserts the
cast; generation under `no_grad` did, raising `expected scalar type Float but found Half`. Generating under
autocast fixes it and has the independent merit of putting eval in the same precision regime as training.
**Revisit if:** loss goes unstable or NaN — measured across four runs it did not, holding at 0.158 with
grad-norm 0.318 — or the student is swapped for a model with a small vocabulary, where the whole trade is
worth far less.

---

## D-026 · Second configuration is rank 16, not rank 32

**Date:** 2026-09-01
**Chose:** LoRA rank 16 as the second configuration, at the same batch 4 x accum 4 as rank 8.
**Over:** the rank 32 the Phase 3 plan named.
**Why:** Rank 32 does not fit. Its adapter is 87M parameters against rank 8's 21.8M, so gradients and
optimiser state add ~900MB on top of a rank-8 run that already peaked at **12.13 of 14.56 GiB**. It trained
55 steps and then met a batch containing one of the long sequences — where the loss materialises a
`batch x seq x 151,936` logits tensor — and asked for **1.93 GiB with 1.86 GiB free**. Short by ~70MB.

The decisive point is what the alternative would have cost. An effective batch of 16 only factors as 4x4 or
2x8, so running rank 32 means dropping to per-device batch 2. That changes fp16 loss scaling and gradient
accumulation dynamics alongside the rank, and the comparison the spec asks for — *a real trade-off between
two configurations* — stops being about rank at all. **Rank 16 at batch 4 is a cleaner rank-vs-rank
comparison than rank 32 at batch 2 could be**, because rank genuinely is the only thing that differs.

Rank 16 also fits the measured envelope: ~43M parameters adds ~450MB over rank 8, leaving ~2GB of margin
against the spike that killed rank 32. The spec's wording is "for example LoRA rank eight against rank
thirty-two", so 8 vs 16 satisfies it.
**Cost, stated plainly:** 8 vs 16 is a 2x capacity spread rather than 4x, so the trade-off it demonstrates
is narrower. If the two land within noise of each other, that is a weaker result than 8 vs 32 would have
given, and the report has to say the spread was constrained by VRAM rather than chosen.
**Revisit if:** training moves to a card with more than 16GB, where rank 32 fits at batch 4 and the original
comparison becomes available.

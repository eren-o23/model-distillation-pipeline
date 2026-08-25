# Project 5 · Fine-Tuning
## Model Distillation Pipeline

**Stack:** Python · Hugging Face · PyTorch · Claude API · W&B · vLLM
**Estimated build time:** 4 to 5 weeks part time

---

### Why this project signals

This is the project that lets you talk about unit economics, which is the fastest way to sound senior. The output is not a model, it is a break-even analysis: the request volume above which owning the model beats renting it. Very few candidates can produce that number from their own work.

---

### Phase 1 — Pick one narrow task a frontier model already does well

- Strong candidates: structured extraction from one document type, classification across many labels, query rewriting, SQL generation against one schema, or PII redaction.
- Weak candidates: open-ended chat and anything requiring broad world knowledge. You are transferring a capability, not creating one.
- Establish the teacher accuracy first on a held-out set. That number is your ceiling, and your entire comparison depends on it being honest.

---

### Phase 2 — Generate a training set from the teacher's best outputs

- Run the teacher over real inputs rather than synthetic prompts, so the training distribution matches the traffic the student will actually see.
- Filter for quality: validate every output against the schema, drop failures rather than teaching them, and deduplicate near-identical inputs.
- Aim for five to ten thousand examples to start. Split train, validation, and test before any tuning, and do not touch the test set until Phase 4.
- This generation run is the main cash cost of the project. Estimate it before you launch it and put the actual number in the README, because cost transparency is part of what you are demonstrating.

---

### Phase 3 — Fine-tune an 8B student

- Use a seven-to-eight-billion-parameter open model with LoRA, or QLoRA if you want it to fit on a single smaller rented GPU.
- Log everything to Weights and Biases: loss curves, learning rate schedule, validation metric per epoch, and sample outputs at each checkpoint.
- Watch for overfitting after two or three epochs on a small dataset, and for degradation if the learning rate is too aggressive.
- Train at least two configurations, for example LoRA rank eight against rank thirty-two, so you have a real trade-off to discuss rather than a single lucky run.

---

### Phase 4 — Benchmark student against teacher on three axes

- **Quality** on the untouched test set using task metrics, plus the judge from Project 4 where the output is open-ended.
- **Cost per thousand requests:** teacher API pricing against student GPU-hour cost amortized across measured throughput, including idle time.
- **Latency** at p50 and p95 for both, measured under comparable concurrency.
- Present it as one table. A typical honest result is the student reaching the mid-nineties percent of teacher quality at a fraction of the cost. If yours underperforms, publish that and explain why — a reported negative result reads as more trustworthy than a suspiciously perfect one.

---

### Phase 5 — Serve it with vLLM and publish the savings

- Serve with vLLM for continuous batching and paged attention. Measure throughput as you raise concurrency and find the point where latency starts degrading.
- Compute the break-even volume: below X requests per day the API is cheaper, above it the self-hosted student wins. This single number is the most senior artifact in the entire portfolio.
- Ship a router in front of it: the student handles the task and escalates to the teacher on low confidence or schema validation failure. Report the escalation rate.

---

### Done when

A three-axis benchmark table, a stated break-even request volume, and a served endpoint with a working escalation path back to the teacher.

**Resume line:** *Distilled a frontier model task into an 8B student at X percent of teacher accuracy and one-twentieth the cost; served with vLLM with break-even at Y requests per day.*

---

### Where this usually goes wrong

- Distilling a task the teacher performs poorly, which caps the student below usable.
- No untouched test set, so every quality claim is contaminated.
- Comparing costs without counting idle GPU time, which flatters the self-hosted option.
- No fallback to the teacher, which makes the system fragile on the tail of the distribution.

**Stretch goals:** Quantize to four bit and remeasure the three axes, collect production data continuously for retraining, and train per-customer adapters swapped at serving time.

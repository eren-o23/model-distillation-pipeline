"""Phase 3: QLoRA fine-tune Qwen3-8B on the teacher-generated training set.

Runs on Kaggle's free 2xT4. Three modes, one script, so the baselines and the fine-tunes are scored by
identical code — a benchmark whose arms run through different harnesses measures the harness.

  python scripts/train_student.py --baseline short          # untuned base, ~60-token prompt
  python scripts/train_student.py --baseline teacher        # untuned base, teacher's ~475-token prompt
  python scripts/train_student.py --rank 8 --limit 256 --epochs 1   # smoke test, reports s/step
  python scripts/train_student.py --rank 8                  # the real run
  python scripts/train_student.py --rank 8 --resume         # after a session is killed
  python scripts/train_student.py --eval-adapter <repo> --eval-n 0  # final full-val eval

Every mode writes reports/raw/phase3/<name>.json. write_phase3.py builds the report from those files, so
no number in it is ever typed by hand.

The test split is not reachable from here: load_split("val") is the only split this script asks for, and
load_split("test") raises without allow_test=True, which appears nowhere in this file.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.data import DATA_DIR, load_split  # noqa: E402
from src.pii.student import SHORT_SYSTEM, render_sft  # noqa: E402
from src.pii.teacher import SYSTEM as TEACHER_SYSTEM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports" / "raw" / "phase3"

# Trainer wraps the model in nn.DataParallel whenever it sees more than one GPU and the process was not
# launched by torchrun. DataParallel re-broadcasts every parameter to every device on each forward, which
# a 4-bit model cannot survive: bitsandbytes weights are not replicable that way, and the broadcast alone
# asked for 2.32GB on top of a resident model — OOM at step 0, before a single step of training.
#
# Pinning the model with device_map is not enough; Trainer counts visible devices, not where the model
# actually sits. So hide the second GPU unless torchrun is genuinely driving the run, in which case each
# rank owns a full replica and DDP is real data parallelism.
if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 4GB was reserved-but-unallocated at the OOM; the allocator fragments badly on the variable-length
# batches dynamic padding produces.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
WANDB_PROJECT = "pii-distillation"
HF_USER = "erenrosman"

# 1024 truncates 0 of 7,842 rows (measured with this tokenizer: mean 276, p95 504, p99 635, max 871).
# It is free because the collator pads to the longest row in each batch, not to this cap — so the common
# ~300-token batch costs ~300 tokens. Truncation would cut the END of the answer, teaching the student to
# emit unterminated JSON, which is precisely the failure Phase 5's router escalates on.
MAX_LENGTH = 1024

# The teacher ceiling was measured on the first 200 val rows. Per-epoch eval uses the same 200 so the
# student's curve and the teacher's number sit on one axis (D-021).
EVAL_N = 200


def build_dataset(tok, path: Path, limit: int | None = None):
    """Tokenise into input_ids + labels, with the prompt masked out to -100.

    Loss is computed on the answer only. Including the prompt would spend most of the gradient teaching
    the model to reproduce a system prompt that is identical in every row and handed to it for free at
    inference: measured on this data, only 33% of tokens are answer, so two thirds of the signal would go
    to copying the question. The mask boundary is asserted token-exact in tests/test_prompt_render.py.
    """
    from datasets import Dataset

    rows = [json.loads(line) for line in path.open()]
    if limit:
        rows = rows[:limit]

    records, truncated = [], 0
    for row in rows:
        prompt, completion = render_sft(tok, row)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > MAX_LENGTH:
            truncated += 1
        ids = (p_ids + c_ids)[:MAX_LENGTH]
        labels = ([-100] * len(p_ids) + c_ids)[:MAX_LENGTH]
        # A row whose prompt alone fills the window contributes no gradient and would quietly dilute the
        # batch. At MAX_LENGTH=1024 this never fires; it is here so that lowering the cap fails loudly.
        assert any(x != -100 for x in labels), f"row {row['messages'][1]['content'][:40]!r} has no target"
        records.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})

    pct = truncated / max(len(records), 1)
    print(f"  {path.name}: {len(records):,} rows, {truncated} truncated ({pct:.2%})")
    assert pct < 0.01, f"{pct:.1%} truncated — raise MAX_LENGTH, a cut answer teaches invalid JSON"
    return Dataset.from_list(records)


def run_eval(model, tok_gen, rows, system, tag: str, wandb_run=None, step=None, batch_size=8) -> dict:
    """Generate, score against gold, log, and persist. Shared by baselines and per-epoch eval."""
    from src.pii.eval import as_dict, evaluate

    # Gradient checkpointing turns the KV cache off; generating without it is slow enough to look like a
    # hang. Restored afterwards so training resumes exactly as configured.
    was_cache, was_training = model.config.use_cache, model.training
    model.config.use_cache = True
    model.eval()
    try:
        t0 = time.time()
        score, samples = evaluate(model, tok_gen, rows, system=system, batch_size=batch_size)
    finally:
        model.config.use_cache = was_cache
        model.train(was_training)

    payload = as_dict(score) | {"tag": tag, "eval_seconds": round(time.time() - t0, 1)}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{tag}.json").write_text(json.dumps(payload | {"samples": samples}, indent=2, ensure_ascii=False))

    print(f"\n[{tag}] micro-F1 {score.micro.f1:.3f}  "
          f"P {score.micro.precision:.3f}  R {score.micro.recall:.3f}  "
          f"schema-invalid {score.schema_invalid}/{score.n_examples}  "
          f"({payload['eval_seconds']}s)", flush=True)
    print(score.table(), flush=True)

    if wandb_run:
        import wandb

        flat = {f"val/{k}": v for k, v in payload.items() if isinstance(v, (int, float))}
        flat |= {f"val_f1/{k}": v["f1"] for k, v in payload["per_label"].items()}
        wandb_run.log(flat, step=step)
        wandb_run.log(
            {"samples": wandb.Table(
                columns=["source_text", "gold", "raw", "parsed"],
                data=[[s["source_text"], json.dumps(s["gold"], ensure_ascii=False),
                       s["raw"], json.dumps(s["parsed"], ensure_ascii=False)] for s in samples],
            )},
            step=step,
        )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, help="LoRA rank; the only variable between the two configs")
    ap.add_argument("--baseline", choices=["short", "teacher"], help="eval the untuned base and exit")
    ap.add_argument("--eval-adapter", help="eval an existing adapter (repo id or local path) and exit")
    ap.add_argument("--eval-n", type=int, default=EVAL_N, help="val rows to score; 0 = all 1,000")
    # 2, not 3 — chosen on measured throughput, see D-024. At 24.44 s/step a 3-epoch run is 10.5h and
    # cannot finish inside Kaggle's 9h session cap; 2 epochs is 7.0h and completes unattended.
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, help="cap training rows, for the smoke test")
    ap.add_argument("--lr", type=float, default=2e-4)
    # Effective batch stays 16 whatever the split. Per-device 4 is affordable once the frozen embedding
    # and output head are back in fp16; before that reclaim, batch 4's backward asked for 1.31GiB the
    # card did not have.
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    ap.add_argument("--out", default="/kaggle/working/phase3")
    ap.add_argument("--no-push", action="store_true", help="skip the HF Hub upload")
    args = ap.parse_args()

    import torch
    from src.pii.eval import load_model, tokenizer

    assert torch.cuda.is_available(), "no GPU visible — check the notebook's accelerator setting"
    cap = torch.cuda.get_device_capability()
    visible = torch.cuda.device_count()
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}), {visible} visible"
          + ("" if visible > 1 else "  [second T4 hidden: DataParallel cannot replicate 4-bit weights]"))
    # bf16 needs Ampere (sm_80+) and so does FlashAttention-2. On a T4 both fail at load, not at step 1 —
    # asserting here turns a confusing traceback into a sentence (D-019).
    assert cap >= (7, 5), f"sm_{cap[0]}{cap[1]} is below the 4-bit requirement"
    if cap < (8, 0):
        print("  sm_75: fp16 + SDPA (bf16 and FlashAttention-2 are Ampere+, and are not used)")

    val_rows = load_split("val")
    eval_rows = val_rows if args.eval_n == 0 else val_rows[: args.eval_n]
    tok_gen = tokenizer()  # left-padded, for generation only

    # ---- eval-only modes -------------------------------------------------------------------------
    if args.baseline or args.eval_adapter:
        system = TEACHER_SYSTEM if args.baseline == "teacher" else None
        tag = f"baseline-{args.baseline}" if args.baseline else f"adapter-{Path(args.eval_adapter).name}"
        if args.eval_n == 0:
            tag += "-full"
        model, _ = load_model(adapter=args.eval_adapter)
        import wandb

        run = wandb.init(project=WANDB_PROJECT, name=tag, job_type="eval",
                         config={"rows": len(eval_rows), "adapter": args.eval_adapter,
                                 "system": "teacher" if system else "short"})
        run_eval(model, tok_gen, eval_rows, system or SHORT_SYSTEM, tag, run)
        run.finish()
        return

    assert args.rank, "pass --rank, --baseline, or --eval-adapter"

    # ---- training --------------------------------------------------------------------------------
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

    name = f"r{args.rank}" + ("-smoke" if args.limit else "")
    out_dir = Path(args.out) / name
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    # A killed Kaggle session that is later --resumed must land back in the SAME W&B run, or the loss
    # curve arrives as two half-runs with no way to see the join. The id is derived from the config, so
    # resuming is automatic and re-running a config deliberately overwrites rather than duplicating.
    os.environ.setdefault("WANDB_RUN_ID", name)
    os.environ.setdefault("WANDB_RESUME", "allow")

    tok_train = tokenizer()
    tok_train.padding_side = "right"  # training pads right; only generation needs left padding
    train_ds = build_dataset(tok_train, DATA_DIR / "train_sft.jsonl", args.limit)

    model, _ = load_model()

    # prepare_model_for_kbit_training upcasts non-quantised parameters to fp32 for fp16 stability. On a
    # 151,936-token vocabulary that is not a rounding error: embed_tokens and lm_head are ~622M params
    # each, so an fp16->fp32 upcast of them costs ~2.5GiB of a 14.56GiB card. Print both sides so the
    # cost is a number rather than a suspicion.
    before = torch.cuda.memory_allocated() / 2**30
    big = {n: (p.dtype, p.numel()) for n, p in model.named_parameters() if p.numel() > 100_000_000}
    # The non-reentrant kwarg has to be passed here too, not only in TrainingArguments: this call enables
    # checkpointing immediately with its own defaults, and the reentrant path cannot see LoRA params as
    # requiring grad.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    after = torch.cuda.memory_allocated() / 2**30
    print(f"  resident {before:.2f} -> {after:.2f} GiB across prepare_model_for_kbit_training")
    for n, (dt, num) in big.items():
        now = dict(model.named_parameters()).get(n)
        print(f"    {n}: {num/1e6:.0f}M  {dt} -> {now.dtype if now is not None else '?'}")

    # Put the giant frozen tensors back in fp16. The blanket fp32 upcast above is correct for layer
    # norms — they are tiny and fp16 training needs the headroom — but Qwen3's 151,936-token vocabulary
    # makes embed_tokens and lm_head ~622M params each, so upcasting them costs 2.32GiB of a 14.56GiB
    # card and buys nothing: LoRA targets only the attention and MLP projections, so neither tensor is
    # trained, holds an optimiser state, or receives a gradient.
    #
    # Selected by size and requires_grad rather than by name, so it cannot silently miss a renamed
    # tensor or catch a trainable one.
    reclaimed = 0
    for p in model.parameters():
        if p.dtype == torch.float32 and p.numel() > 100_000_000 and not p.requires_grad:
            p.data = p.data.to(torch.float16)
            reclaimed += p.numel() * 2
    if reclaimed:
        torch.cuda.empty_cache()
        print(f"  recast {reclaimed / 2**30:.2f} GiB of frozen fp32 back to fp16 · resident now "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    # alpha = 2*rank holds the alpha/r scaling constant across the two configs, so "rank 8 vs 32" varies
    # adapter capacity alone instead of confounding it with a 4x change in effective update size (D-020).
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    # Enable explicitly AFTER the PEFT wrap, and then check rather than assume. Three separate places
    # ask for checkpointing (prepare_model_for_kbit_training, this call, TrainingArguments) and it is
    # still possible for none of them to stick through the wrap — the symptom is not an error but ~7GB
    # of stored activations and an OOM on the second step, which reads as "the T4 is too small".
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing, "gradient checkpointing is off — an 8B will not fit without it"
    print(f"  gradient checkpointing: on · weights resident {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    # Peak is measured per phase. A single figure spanning the whole process cannot say whether the
    # ceiling came from training or from the generation eval, and those have opposite fixes: a smaller
    # training batch, versus a smaller eval batch.
    torch.cuda.reset_peak_memory_stats()

    # warmup_ratio is deprecated in transformers 5.x. Computing the step count explicitly also makes the
    # schedule legible in the log rather than implied.
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch * args.accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(0.03 * total_steps))
    print(f"  {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps "
          f"(effective batch {args.batch * args.accum}, {warmup_steps} warmup)")

    best = {"f1": -1.0, "epoch": None}
    eval_seconds: list[float] = []

    class EvalEachEpoch(TrainerCallback):
        """Per-epoch val score against gold, plus the adapter that produced it.

        Selection is by micro-F1, not by val loss: they disagree, and F1 is the number the project
        reports. Saving each epoch's adapter separately is also what makes a killed Kaggle session cost
        one epoch rather than the whole run.
        """

        def on_epoch_end(self, targs, state, control, model=None, **kw):
            import wandb

            # Under torchrun every rank runs this callback. Without the guard the ranks would duplicate
            # the generation work and race each other writing the same JSON file.
            if not state.is_world_process_zero:
                return

            epoch = round(state.epoch)
            adapter_dir = out_dir / f"epoch{epoch}"
            model.save_pretrained(adapter_dir)

            train_peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            # Measured: eval peaks at 6.84GiB against training's 12.13, so the card is mostly idle here.
            # Generation is latency-bound on a T4, and a wider batch is close to free — at 32 rows a
            # batch of 4 took 202.9s, which at the default 200 rows would be 21 minutes per epoch.
            payload = run_eval(model, tok_gen, eval_rows, SHORT_SYSTEM, f"{name}-epoch{epoch}",
                               wandb_run=wandb.run, step=state.global_step, batch_size=16)
            eval_seconds.append(payload["eval_seconds"])
            print(f"  peak GiB — training {train_peak:.2f}, eval "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f}, of "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            if payload["micro_f1"] > best["f1"]:
                best.update(f1=payload["micro_f1"], epoch=epoch, dir=str(adapter_dir))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            fp16=True,          # bf16 is unavailable on sm_75 (D-019)
            bf16=False,
            gradient_checkpointing=True,
            # Reentrant checkpointing does not see the LoRA params as requiring grad, and the run fails
            # with "none of the inputs have requires_grad" rather than training badly.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # NOT paged. Paged optimizers keep their state in CUDA unified memory so it can spill to host
            # RAM under pressure — worth it when the optimizer state is large, pointless for LoRA where
            # it is ~350MB even at rank 32. Rank 32 died 71% into epoch 1 with an illegal memory access
            # inside the paged optimizer's sync_gpu; rank 8 survived only because its state was small
            # enough that paging never engaged. Removing the paging removes the fault path entirely.
            optim="adamw_8bit",
            logging_steps=10,
            # Checkpoint on steps, not epochs. The rank-32 crash landed 71% into epoch 1, so an
            # epoch-only strategy had written nothing and 2.6h of GPU was unrecoverable. LoRA
            # checkpoints are adapter-sized, so 250 steps (~1.8h apart at 26.5 s/step) is cheap
            # insurance that makes --resume able to actually save a run.
            save_strategy="steps",
            save_steps=250,
            save_total_limit=2,  # /kaggle/working caps at 20GB
            report_to="wandb",
            run_name=name,
            seed=20260825,       # the same seed the splits are frozen with
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tok_train, padding=True, label_pad_token_id=-100),
        callbacks=[EvalEachEpoch()],
    )

    result = trainer.train(resume_from_checkpoint=args.resume or None)

    # The smoke test's real output. Projected against the FULL dataset and 3 epochs regardless of what
    # this run did, because that is the number that decides single-GPU vs torchrun DDP.
    # train_runtime includes the epoch-end eval callback, which on a smoke run is most of it. Subtract
    # it, or the projection is dominated by a cost that does not scale with dataset size.
    eval_total = sum(eval_seconds)
    train_only = result.metrics["train_runtime"] - eval_total
    sec_per_step = train_only / max(trainer.state.global_step, 1)

    n_rows = sum(1 for _ in (DATA_DIR / "train_sft.jsonl").open())
    per_epoch_s = sec_per_step * n_rows / (args.batch * args.accum)
    per_eval_s = (eval_total / len(eval_seconds)) if eval_seconds else 0.0
    print(f"\n{sec_per_step:.2f} s/step training over {trainer.state.global_step} steps "
          f"(eval excluded: {eval_total:.0f}s across {len(eval_seconds)})", flush=True)
    for n_ep in (2, 3):
        total = (per_epoch_s * n_ep + per_eval_s * n_ep) / 3600
        fits = "fits a 9h session" if total < 8.5 else "EXCEEDS a 9h session"
        print(f"  full run, {n_ep} epochs: {total:.1f}h  ({fits})", flush=True)

    print(f"\nbest epoch: {best['epoch']} at micro-F1 {best['f1']:.3f}")
    (RAW / f"{name}-summary.json").write_text(json.dumps(
        {"config": vars(args), "best": best, "sec_per_step": sec_per_step,
         "train_runtime_s": result.metrics["train_runtime"]}, indent=2))

    if not args.no_push and not args.limit and best.get("dir"):
        from huggingface_hub import HfApi

        repo = f"{HF_USER}/pii-qwen3-8b-lora-r{args.rank}"
        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=best["dir"], repo_id=repo)
        print(f"pushed epoch {best['epoch']} adapter to https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()

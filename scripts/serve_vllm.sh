#!/usr/bin/env bash
# Phase 5: merge the adapter into the base and serve it with vLLM. Runs on the rented 24GB box.
#
#   bash scripts/serve_vllm.sh                                    # defaults to the deployed rank-8 adapter
#   ADAPTER=erenrosman/pii-qwen3-8b-lora-r16 bash scripts/serve_vllm.sh
#
# The box is ephemeral and will be re-provisioned more than once, which is the only reason this is a
# script rather than three commands in a README.
set -euo pipefail

ADAPTER=${ADAPTER:-erenrosman/pii-qwen3-8b-lora-r8}
MERGED=${MERGED:-$HOME/pii-student-merged}
SERVED_NAME=${SERVED_NAME:-pii-student}
PORT=${PORT:-8000}

# Merging removes LoRA-at-serving-time as a variable in the throughput number. vLLM can serve the
# adapter directly with --enable-lora, but then every measurement carries an extra moving part that
# Phase 4's baseline did not have. Merge once, serve plain weights.
#
# Note this produces an fp16 model where Phase 4 measured 4-bit NF4 + adapter. That is a real model
# change, which is why benchmark_vllm.py --parity is a gate and not a formality.
if [ ! -d "$MERGED" ]; then
  echo "merging $ADAPTER into Qwen/Qwen3-8B (fp16, on CPU — the GPU is for serving) …"
  python - <<PYEOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(base, "$ADAPTER").merge_and_unload()
merged.save_pretrained("$MERGED")
# The tokenizer must travel with the weights: render_prompt renders against it, and a server loading
# a different tokenizer revision would shift the prompt out from under the fine-tune.
AutoTokenizer.from_pretrained("Qwen/Qwen3-8B").save_pretrained("$MERGED")
print("merged ->", "$MERGED")
PYEOF
else
  echo "reusing existing merge at $MERGED"
fi

# --max-model-len 2048, not 1024: prompts reach ~871 tokens and generation is capped at 512
# (eval.MAX_NEW_TOKENS), so 1024 would silently truncate the longest requests and read as the student
# failing schema validation. Everything above 1383 is headroom; 2048 is the next sensible stop.
exec vllm serve "$MERGED" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT"

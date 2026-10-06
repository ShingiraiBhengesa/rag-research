#!/bin/bash
# Launch one OpenAI-compatible vLLM server for the adaptive-cue pilot.
#
# TEMPLATE: the <<...>> values and the conda/env paths must be verified inside
# the NRP pod. Victim serving flags mirror canonical production
# (attack/launch/generate_production_arrays.py). Precision/quantization are
# fixed to bfloat16 and must not be lowered to fit memory -- stage execution
# instead (see README.md).
set -euo pipefail

ROLE="${1:?usage: vllm_serve.sh <victim|attacker> <model_key>}"
MODEL_KEY="${2:?model key: qwen7b|qwen14b|llama8b}"

case "$MODEL_KEY" in
  qwen7b)  MODEL_ID="Qwen/Qwen2.5-7B-Instruct";        REVISION="a09a35458c702b33eeacc393d103063234e8bc28"; PARSER="hermes" ;;
  qwen14b) MODEL_ID="Qwen/Qwen2.5-14B-Instruct";       REVISION="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"; PARSER="hermes" ;;
  llama8b) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"; REVISION="0e9e39f249a16976918f6564b8830bc894c89659"; PARSER="llama3_json" ;;
  *) echo "unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.84}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# The attacker uses the same serving flags as a qwen victim; tool-choice is
# harmless for the attacker (it uses guided_json, not tool calls).
echo "serving role=$ROLE model=$MODEL_ID revision=$REVISION parser=$PARSER port=$PORT"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --revision "$REVISION" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser "$PARSER" \
  --enforce-eager

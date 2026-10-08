#!/bin/bash
# Infrastructure-only driver for the dynamic bridge-cue pilot. Used unchanged by
# the NRP Jobs (k8s/) and the Libra smoke sbatch (../slurm/). It serves models
# with the canonical production vLLM flags (bf16, no quantization) and calls
# the pilot CLI; it contains no scientific logic.
#
#   run_phase.sh decompose <split>
#   run_phase.sh cell <split> <victim> <dataset>
#
# Required env: REPO, PY_ODR (driver python; runs the victim ODR graph),
# PY_VLLM (serving python), STATE_DIR (checkpoints), RESULTS_DIR (analysis
# outputs), HF_HOME. Optional PY_CUES: python with torch+transformers for the
# cue phase's NLI check (default PY_ODR; on Libra the odr env lacks torch, so
# the smoke sbatch points PY_CUES at the vllm env).
# Optional: SMOKE_PROVISIONAL_LAMBDA=1 (smoke only), MAX_QUESTIONS (smoke/dev only).
#
# Preemption: every CLI phase checkpoints per unit; a phase marker is written
# only when the phase exits 0 with no new infrastructure failures, so a
# restarted pod skips finished phases (and their model loads) entirely.
set -euo pipefail

MODE="${1:?mode: decompose|cell}"; SPLIT="${2:?split}"
VICTIM="${3:-}"; DATASET="${4:-}"
: "${REPO:?}" "${PY_ODR:?}" "${PY_VLLM:?}" "${STATE_DIR:?}" "${RESULTS_DIR:?}" "${HF_HOME:?}"
cd "$REPO"
PY_CUES="${PY_CUES:-$PY_ODR}"
export PYTHONPATH="$REPO:$REPO/open_deep_research/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
MARKERS="$STATE_DIR/$SPLIT/markers"; mkdir -p "$MARKERS" "$STATE_DIR/logs"
PORT="${PORT:-8000}"; URL="http://127.0.0.1:$PORT/v1"
SERVER_PID=""
COMMON=(--split "$SPLIT" --state-dir "$STATE_DIR" --results-dir "$RESULTS_DIR")
[[ -n "${MAX_QUESTIONS:-}" ]] && COMMON+=(--max-questions "$MAX_QUESTIONS")

model_args() {  # key -> "model_id revision parser"; mirrors attack/dynamic_bridge_pilot/config.py
  case "$1" in
    qwen14b) echo "Qwen/Qwen2.5-14B-Instruct cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 hermes" ;;
    llama8b) echo "meta-llama/Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659 llama3_json" ;;
    *) echo "unknown model key $1" >&2; exit 2 ;;
  esac
}

serve() {
  stop_server
  read -r MID REV PARSER < <(model_args "$1")
  echo "[serve] $1 $MID@$REV"
  "$PY_VLLM" -m vllm.entrypoints.openai.api_server \
    --model "$MID" --revision "$REV" --served-model-name "$MID" --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" --max-model-len 32768 \
    --tensor-parallel-size 1 --generation-config vllm --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" --enforce-eager > "$STATE_DIR/logs/vllm_${1}_$(date +%s).log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "[serve] ready"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[serve] vLLM died" >&2; exit 1; }
    sleep 10
  done
  echo "[serve] timeout" >&2; exit 1
}
stop_server() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; true; }
trap stop_server EXIT

n_failures() { cat "$STATE_DIR/$SPLIT/failures/"*.jsonl 2>/dev/null | wc -l; }

phase() {  # phase <marker-name> <cli args...>
  local name="$1"; shift
  [[ -f "$MARKERS/$name.done" ]] && { echo "[skip] $name"; return 0; }
  local before; before=$(n_failures)
  "${PHASE_PY:-$PY_ODR}" -B -m attack.dynamic_bridge_pilot.cli "$@" "${COMMON[@]}"
  if [[ "$(n_failures)" == "$before" ]]; then date -u +%FT%TZ > "$MARKERS/$name.done"
  else echo "[phase] $name had infrastructure failures; not marking done (rerun retries them)" >&2; exit 1; fi
}

if [[ "$MODE" == "decompose" ]]; then
  [[ -f "$MARKERS/decompose.done" ]] && { echo "[skip] decompose"; exit 0; }
  serve qwen14b
  phase decompose decompose --attacker-base-url "$URL"
  exit 0
fi

: "${VICTIM:?victim}" "${DATASET:?dataset}"
CELL="${VICTIM}_${DATASET}"
until [[ -f "$MARKERS/decompose.done" ]]; do echo "[wait] decompositions for $SPLIT"; sleep 60; done

SELECT_ARGS=()
if [[ "$SPLIT" == "smoke" && "${SMOKE_PROVISIONAL_LAMBDA:-0}" == "1" ]]; then SELECT_ARGS+=(--provisional-equal-lambda); fi
if [[ "$SPLIT" == "heldout" ]]; then "$PY_ODR" -B -m attack.dynamic_bridge_pilot.cli verify-freeze --split heldout; fi

if [[ "$VICTIM" == "qwen14b" ]]; then
  # Attacker and victim share Qwen2.5-14B weights: one server for all phases.
  need=0; for p in anchor cues arms; do [[ -f "$MARKERS/${p}_$CELL.done" ]] || need=1; done
  [[ $need == 1 ]] && serve qwen14b
  phase "anchor_$CELL" anchor --victim qwen14b --datasets "$DATASET" --victim-base-url "$URL"
  PHASE_PY="$PY_CUES" phase "cues_$CELL" cues --victim qwen14b --datasets "$DATASET" --attacker-base-url "$URL"
  [[ "$SPLIT" != "dev" ]] && phase "select_$CELL" select --victim qwen14b --datasets "$DATASET" "${SELECT_ARGS[@]}"
  [[ "$SPLIT" != "dev" ]] && phase "arms_$CELL"   arms   --victim qwen14b --datasets "$DATASET" --victim-base-url "$URL"
else
  # One 48 GB L40 cannot hold the 8B victim and the 14B attacker at bf16 at once:
  # stage the model loads (victim -> attacker -> victim). Same code, same order of units.
  [[ -f "$MARKERS/anchor_$CELL.done" ]] || { serve "$VICTIM"; phase "anchor_$CELL" anchor --victim "$VICTIM" --datasets "$DATASET" --victim-base-url "$URL"; }
  [[ -f "$MARKERS/cues_$CELL.done" ]] || { serve qwen14b; PHASE_PY="$PY_CUES" phase "cues_$CELL" cues --victim "$VICTIM" --datasets "$DATASET" --attacker-base-url "$URL"; }
  stop_server
  if [[ "$SPLIT" != "dev" ]]; then
    phase "select_$CELL" select --victim "$VICTIM" --datasets "$DATASET" "${SELECT_ARGS[@]}"
    [[ -f "$MARKERS/arms_$CELL.done" ]] || { serve "$VICTIM"; phase "arms_$CELL" arms --victim "$VICTIM" --datasets "$DATASET" --victim-base-url "$URL"; }
  fi
fi
echo "[done] $SPLIT $CELL"

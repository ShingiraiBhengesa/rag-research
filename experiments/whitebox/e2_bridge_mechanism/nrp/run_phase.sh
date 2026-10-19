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
# Optional: SERVE_ENV_SCRIPT (sourced around the vLLM server only), SMOKE_PROVISIONAL_LAMBDA=1 (smoke only), MAX_QUESTIONS (smoke/dev only),
# STOP_AFTER=anchor (cell mode: run only the clean anchor runs, e.g. to evaluate
# the baseline gate before any cue or arm work), NUM_SHARDS (cell mode; shard index from SHARD or the Indexed Job's
# JOB_COMPLETION_INDEX). Shard i takes every NUM_SHARDS-th question and writes
# its own checkpoint files and markers.
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
export PYTHONPATH="$REPO/src:$REPO/third_party/open_deep_research/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
MARKERS="$STATE_DIR/$SPLIT/markers"; mkdir -p "$MARKERS" "$STATE_DIR/logs"
# Random free port (as in the canonical sbatch): GPU nodes are shared with other vLLM jobs.
PORT="${PORT:-$("$PY_VLLM" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
URL="http://127.0.0.1:$PORT/v1"
SERVER_PID=""
COMMON=(--split "$SPLIT" --state-dir "$STATE_DIR" --results-dir "$RESULTS_DIR")
[[ -n "${MAX_QUESTIONS:-}" ]] && COMMON+=(--max-questions "$MAX_QUESTIONS")

model_args() {  # key -> "model_id revision parser"; mirrors src/prematurestop/pipelines/bridge_mechanism/config.py
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
  # Optional SERVE_ENV_SCRIPT is sourced in a subshell around the server only
  # (Libra needs the canonical system-GCC toolchain for Triton JIT; NRP doesn't).
  # VLLM_SERVER_DEV_MODE=1 exposes POST /reset_prefix_cache, which the pilot calls
  # before every victim run (config.RESET_PREFIX_CACHE).
  ( export VLLM_SERVER_DEV_MODE=1
    if [[ -n "${SERVE_ENV_SCRIPT:-}" ]]; then source "$SERVE_ENV_SCRIPT"; fi
    exec "$PY_VLLM" -m vllm.entrypoints.openai.api_server \
    --model "$MID" --revision "$REV" --served-model-name "$MID" --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" --max-model-len 32768 \
    --tensor-parallel-size 1 --generation-config vllm --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" --enforce-eager ) > "$STATE_DIR/logs/vllm_${1}_$(date +%s).log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "[serve] ready"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[serve] vLLM died" >&2; exit 1; }
    sleep 10
  done
  echo "[serve] timeout" >&2; exit 1
}
# `wait` on a server we just killed returns 143; under set -e that must not end the script.
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap 'rc=$?; stop_server; exit $rc' EXIT

# `|| true` inside the group: with pipefail, a missing failures/ glob would
# otherwise fail the pipeline and silently kill the script under set -e.
n_failures() { { cat "$STATE_DIR/$SPLIT/failures/"*.jsonl 2>/dev/null || true; } | wc -l; }

phase() {  # phase <marker-name> <cli args...>
  local name="$1"; shift
  [[ -f "$MARKERS/$name.done" ]] && { echo "[skip] $name"; return 0; }
  local before; before=$(n_failures)
  "${PHASE_PY:-$PY_ODR}" -B -m prematurestop.pipelines.bridge_mechanism.cli "$@" "${COMMON[@]}"
  if [[ "$(n_failures)" == "$before" ]]; then date -u +%FT%TZ > "$MARKERS/$name.done"
  else echo "[phase] $name had infrastructure failures; not marking done (rerun retries them)" >&2; exit 1; fi
}

NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"

if [[ "$MODE" == "decompose" ]]; then
  [[ -f "$MARKERS/decompose.done" ]] && { echo "[skip] decompose"; exit 0; }
  serve qwen14b
  phase decompose decompose --attacker-base-url "$URL"
  exit 0
fi

: "${VICTIM:?victim}" "${DATASET:?dataset}"
CELL="${VICTIM}_${DATASET}"
if (( NUM_SHARDS > 1 )); then CELL="${CELL}.s$(printf %03d "$SHARD")of$(printf %03d "$NUM_SHARDS")"; fi
echo "[cell] $SPLIT $VICTIM $DATASET shard $SHARD/$NUM_SHARDS"
until [[ -f "$MARKERS/decompose.done" ]]; do echo "[wait] decompositions for $SPLIT"; sleep 60; done

SELECT_ARGS=()
if [[ "$SPLIT" == "smoke" && "${SMOKE_PROVISIONAL_LAMBDA:-0}" == "1" ]]; then SELECT_ARGS+=(--provisional-equal-lambda); fi
# dev: anchor + cues always; select + arms only once lambda is frozen (the
# freeze is committed, so the post-freeze commit re-runs this cell and picks
# up where the pre-freeze commit stopped). full: everything, freeze verified first.
RUN_ARMS=1
if [[ "$SPLIT" == "dev" && ! -f "$REPO/experiments/whitebox/e2_bridge_mechanism/freeze_manifest.json" ]]; then RUN_ARMS=0; fi
if [[ "$SPLIT" == "full" ]]; then "$PY_ODR" -B -m prematurestop.pipelines.bridge_mechanism.cli verify-freeze --split full; fi
export PILOT_SHARD="$SHARD" PILOT_NUM_SHARDS="$NUM_SHARDS"   # read by the cell phases below

if [[ "$VICTIM" == "qwen14b" ]]; then
  # Attacker and victim share Qwen2.5-14B weights: one server for all phases.
  need=0; for p in anchor cues $([[ $RUN_ARMS == 1 ]] && echo arms); do [[ -f "$MARKERS/${p}_$CELL.done" ]] || need=1; done
  [[ $need == 1 ]] && serve qwen14b
  phase "anchor_$CELL" anchor --victim qwen14b --datasets "$DATASET" --victim-base-url "$URL"
  [[ "${STOP_AFTER:-}" == anchor ]] && { echo "[done] $SPLIT $CELL anchors only (STOP_AFTER=anchor)"; exit 0; }
  PHASE_PY="$PY_CUES" phase "cues_$CELL" cues --victim qwen14b --datasets "$DATASET" --attacker-base-url "$URL"
  if [[ $RUN_ARMS == 1 ]]; then
    phase "select_$CELL" select --victim qwen14b --datasets "$DATASET" "${SELECT_ARGS[@]}"
    phase "arms_$CELL"   arms   --victim qwen14b --datasets "$DATASET" --victim-base-url "$URL"
  fi
else
  # One 48 GB L40 cannot hold the 8B victim and the 14B attacker at bf16 at once:
  # stage the model loads (victim -> attacker -> victim). Same code, same order of units.
  [[ -f "$MARKERS/anchor_$CELL.done" ]] || { serve "$VICTIM"; phase "anchor_$CELL" anchor --victim "$VICTIM" --datasets "$DATASET" --victim-base-url "$URL"; }
  [[ "${STOP_AFTER:-}" == anchor ]] && { stop_server; echo "[done] $SPLIT $CELL anchors only (STOP_AFTER=anchor)"; exit 0; }
  [[ -f "$MARKERS/cues_$CELL.done" ]] || { serve qwen14b; PHASE_PY="$PY_CUES" phase "cues_$CELL" cues --victim "$VICTIM" --datasets "$DATASET" --attacker-base-url "$URL"; }
  stop_server
  if [[ $RUN_ARMS == 1 ]]; then
    phase "select_$CELL" select --victim "$VICTIM" --datasets "$DATASET" "${SELECT_ARGS[@]}"
    [[ -f "$MARKERS/arms_$CELL.done" ]] || { serve "$VICTIM"; phase "arms_$CELL" arms --victim "$VICTIM" --datasets "$DATASET" --victim-base-url "$URL"; }
  fi
fi
[[ $RUN_ARMS == 1 ]] && echo "[done] $SPLIT $CELL" || echo "[done] $SPLIT $CELL cues only (lambda not frozen yet)"

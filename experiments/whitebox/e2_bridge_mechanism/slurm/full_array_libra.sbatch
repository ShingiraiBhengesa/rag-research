#!/bin/bash
# All four victim x dataset cells of a split as ONE array (4 cells x NUM_SHARDS
# shards), throttled so tasks run a few at a time on their own. Task id t maps
# to cell t / NUM_SHARDS (order below) and shard t % NUM_SHARDS. Run from a
# pinned worktree, e.g.
#   SPLIT=full NUM_SHARDS=8 STOP_AFTER=anchor sbatch --array=0-31%4 .../full_array_libra.sbatch
#   SPLIT=full NUM_SHARDS=8 STOP_AFTER= sbatch --array=0-31%4 --dependency=aftercorr:<anchor array> .../full_array_libra.sbatch
# Change the throttle later with: scontrol update JobId=<array id> ArrayTaskThrottle=N
#SBATCH --job-name=dbcp_full
#SBATCH --partition=ccstar
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --nodelist=ml01,ml02
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=local_data/logs/e2_bridge_mechanism/%x_%A_%a.out
#SBATCH --error=local_data/logs/e2_bridge_mechanism/%x_%A_%a.err
set -euo pipefail
: "${SPLIT:?}" "${NUM_SHARDS:?}"
CELLS=(qwen14b:hotpot qwen14b:2wiki llama8b:hotpot llama8b:2wiki)
T="${SLURM_ARRAY_TASK_ID:?run as an array}"
CELL="${CELLS[$(( T / NUM_SHARDS ))]}"
export VICTIM="${CELL%%:*}" DATASET="${CELL##*:}" SHARD=$(( T % NUM_SHARDS ))
export REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PY_ODR=$HOME/.conda/envs/odr/bin/python
export PY_VLLM=$HOME/.conda/envs/vllm/bin/python
export PY_CUES=$HOME/.conda/envs/vllm/bin/python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export STATE_DIR=${PREMATURESTOP_LOCAL_DATA:-$REPO/local_data}/e2_bridge_mechanism/state
export RESULTS_DIR=${PREMATURESTOP_LOCAL_DATA:-$REPO/local_data}/e2_bridge_mechanism/run_results
export SERVE_ENV_SCRIPT="$REPO/experiments/whitebox/e2_bridge_mechanism/slurm/libra_vllm_toolchain.sh"
echo "[full-array] commit $(git -C "$REPO" rev-parse HEAD) node $(hostname) task $T -> $SPLIT $VICTIM/$DATASET shard $SHARD/$NUM_SHARDS stop_after=${STOP_AFTER:-}"
bash "$REPO/experiments/whitebox/e2_bridge_mechanism/nrp/run_phase.sh" cell "$SPLIT" "$VICTIM" "$DATASET"

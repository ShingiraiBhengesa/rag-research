#!/bin/bash
# One victim x dataset cell (optionally one shard) on one Libra H100, through the
# same driver the NRP Jobs use. Run from a pinned worktree of a pushed commit:
#   SPLIT=dev VICTIM=qwen14b DATASET=hotpot sbatch .../cell_libra.sbatch
#   SPLIT=full VICTIM=llama8b DATASET=2wiki NUM_SHARDS=8 sbatch --array=0-7 .../cell_libra.sbatch
# (array task id = shard). The decompose marker must already exist for SPLIT.
#SBATCH --job-name=dbcp_cell
#SBATCH --partition=ccstar
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --nodelist=ml01,ml02
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%A_%a.out
#SBATCH --error=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%A_%a.err
set -euo pipefail
: "${SPLIT:?}" "${VICTIM:?}" "${DATASET:?}"
export REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PY_ODR=/home/sbhengesa/.conda/envs/odr/bin/python
export PY_VLLM=/home/sbhengesa/.conda/envs/vllm/bin/python
export PY_CUES=/home/sbhengesa/.conda/envs/vllm/bin/python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export STATE_DIR=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot/state
export RESULTS_DIR=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot/run_results
export SERVE_ENV_SCRIPT="$REPO/experiments/dynamic_bridge_cue_pilot/slurm/libra_vllm_toolchain.sh"
export NUM_SHARDS="${NUM_SHARDS:-1}" SHARD="${SLURM_ARRAY_TASK_ID:-${SHARD:-0}}"
echo "[cell] commit $(git -C "$REPO" rev-parse HEAD) node $(hostname) $SPLIT $VICTIM/$DATASET shard $SHARD/$NUM_SHARDS"
bash "$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh" cell "$SPLIT" "$VICTIM" "$DATASET"

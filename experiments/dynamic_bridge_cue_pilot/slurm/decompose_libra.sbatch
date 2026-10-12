#!/bin/bash
# Step 1 (decomposition) for a split on one Libra H100. Run from a pinned worktree:
#   SPLIT=full sbatch .../decompose_libra.sbatch
#SBATCH --job-name=dbcp_decomp
#SBATCH --partition=ccstar
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --nodelist=ml01,ml02
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%j.err
set -euo pipefail
: "${SPLIT:?}"
export REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PY_ODR=/home/sbhengesa/.conda/envs/odr/bin/python
export PY_VLLM=/home/sbhengesa/.conda/envs/vllm/bin/python
export PY_CUES=/home/sbhengesa/.conda/envs/vllm/bin/python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export STATE_DIR=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot/state
export RESULTS_DIR=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot/run_results
export SERVE_ENV_SCRIPT="$REPO/experiments/dynamic_bridge_cue_pilot/slurm/libra_vllm_toolchain.sh"
echo "[decompose] commit $(git -C "$REPO" rev-parse HEAD) node $(hostname) split $SPLIT"
bash "$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh" decompose "$SPLIT"

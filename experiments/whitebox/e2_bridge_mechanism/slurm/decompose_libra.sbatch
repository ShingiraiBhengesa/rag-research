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
#SBATCH --output=local_data/logs/e2_bridge_mechanism/%x_%j.out
#SBATCH --error=local_data/logs/e2_bridge_mechanism/%x_%j.err
set -euo pipefail
: "${SPLIT:?}"
export REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PY_ODR=$HOME/.conda/envs/odr/bin/python
export PY_VLLM=$HOME/.conda/envs/vllm/bin/python
export PY_CUES=$HOME/.conda/envs/vllm/bin/python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export STATE_DIR=${PREMATURESTOP_LOCAL_DATA:-$REPO/local_data}/e2_bridge_mechanism/state
export RESULTS_DIR=${PREMATURESTOP_LOCAL_DATA:-$REPO/local_data}/e2_bridge_mechanism/run_results
export SERVE_ENV_SCRIPT="$REPO/experiments/whitebox/e2_bridge_mechanism/slurm/libra_vllm_toolchain.sh"
echo "[decompose] commit $(git -C "$REPO" rev-parse HEAD) node $(hostname) split $SPLIT"
bash "$REPO/experiments/whitebox/e2_bridge_mechanism/nrp/run_phase.sh" decompose "$SPLIT"

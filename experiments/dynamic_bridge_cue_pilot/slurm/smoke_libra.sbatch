#!/bin/bash
# Smoke test on Libra (decision 5): 3 questions x 2 datasets x 2 victims x 6 arms
# = 72 victim arm runs + 12 anchor runs, on one H100, sequentially, through the
# SAME infrastructure driver the NRP Jobs use (nrp/run_phase.sh). Provisional
# equal lambda (smoke only). Submit only after explicit go-ahead:
#   sbatch experiments/dynamic_bridge_cue_pilot/slurm/smoke_libra.sbatch
#SBATCH --job-name=dbcp_smoke
#SBATCH --partition=ccstar
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --nodelist=ml01,ml02
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/dynamic_bridge_cue_pilot/%x_%j.out
#SBATCH --error=results/logs/dynamic_bridge_cue_pilot/%x_%j.err
set -euo pipefail
export REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PY_ODR=/home/sbhengesa/.conda/envs/odr/bin/python
export PY_VLLM=/home/sbhengesa/.conda/envs/vllm/bin/python
export PY_CUES=/home/sbhengesa/.conda/envs/vllm/bin/python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export STATE_DIR="$REPO/experiments/dynamic_bridge_cue_pilot/state"
export RESULTS_DIR="$REPO/experiments/dynamic_bridge_cue_pilot/smoke_results"
export SMOKE_PROVISIONAL_LAMBDA=1
D="$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh"
bash "$D" decompose smoke
for v in qwen14b llama8b; do for d in hotpot 2wiki; do bash "$D" cell smoke "$v" "$d"; done; done
# CPU analysis in the whitebox env (matplotlib/pandas)
PYTHONPATH="$REPO:$REPO/open_deep_research/src" /home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m attack.dynamic_bridge_pilot.cli analyze --split smoke --state-dir "$STATE_DIR" --results-dir "$RESULTS_DIR"

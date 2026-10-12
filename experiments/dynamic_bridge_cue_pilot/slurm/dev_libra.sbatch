#!/bin/bash
# Dev stage 1 on Libra: decompose + clean anchor runs + cue generation for the
# 20 dev questions, both victims, both datasets, on ONE H100, through the same
# driver the NRP Jobs use. With no freeze manifest in the checkout, the cells
# stop after cue generation (no victim arms yet). Then: baseline gate on the dev
# anchors and the cue-rating sheet for lambda.
#
# Run from a git worktree at a pushed commit (REPO = that worktree), e.g.
#   git worktree add ~/dbcp_runs/<sha> <sha>
#   cd ~/dbcp_runs/<sha> && sbatch experiments/dynamic_bridge_cue_pilot/slurm/dev_libra.sbatch
#SBATCH --job-name=dbcp_dev
#SBATCH --partition=ccstar
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --nodelist=ml01,ml02
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%j.err
set -euo pipefail
export REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PY_ODR=/home/sbhengesa/.conda/envs/odr/bin/python
export PY_VLLM=/home/sbhengesa/.conda/envs/vllm/bin/python
export PY_CUES=/home/sbhengesa/.conda/envs/vllm/bin/python
PY_AN=/home/sbhengesa/.conda/envs/whitebox/bin/python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export STATE_DIR=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot/state
export RESULTS_DIR=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot/run_results
echo "[dev] commit $(git -C "$REPO" rev-parse HEAD)  node $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-?}"
D="$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh"
bash "$D" decompose dev
for v in qwen14b llama8b; do for d in hotpot 2wiki; do bash "$D" cell dev "$v" "$d"; done; done
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/open_deep_research/src" HF_HUB_OFFLINE=1
"$PY_AN" -B -m attack.dynamic_bridge_pilot.cli baseline-gate --split dev --state-dir "$STATE_DIR" --results-dir "$RESULTS_DIR" || echo "[dev] baseline gate: FAIL in at least one cell (see baseline_gate.csv)"
"$PY_AN" -B -m attack.dynamic_bridge_pilot.cli audit-export --split dev --state-dir "$STATE_DIR" --out "$RESULTS_DIR/dev/dev_cue_ratings_blank.csv"
echo "[dev] stage 1 complete"

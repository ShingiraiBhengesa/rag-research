#!/bin/bash
# CPU-only analysis of a split (labels, PCR, primary/secondary outcomes, gate,
# replication check, figures, summary). Run from the repo or a pinned worktree:
#   SPLIT=full sbatch experiments/dynamic_bridge_cue_pilot/slurm/analyze_libra.sbatch
#SBATCH --job-name=dbcp_analyze
#SBATCH --partition=ccstar
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/results/logs/dynamic_bridge_cue_pilot/%x_%j.err
set -euo pipefail
: "${SPLIT:?}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/open_deep_research/src" HF_HUB_OFFLINE=1
S=/home/sbhengesa/rag-research/experiments/dynamic_bridge_cue_pilot
echo "[analyze] commit $(git rev-parse HEAD) node $(hostname) split $SPLIT"
/home/sbhengesa/.conda/envs/whitebox/bin/python -B -m attack.dynamic_bridge_pilot.cli verify-freeze
/home/sbhengesa/.conda/envs/whitebox/bin/python -B -m attack.dynamic_bridge_pilot.cli analyze --split "$SPLIT" \
  --state-dir "$S/state" --results-dir "$S/run_results"
echo "[analyze] done"

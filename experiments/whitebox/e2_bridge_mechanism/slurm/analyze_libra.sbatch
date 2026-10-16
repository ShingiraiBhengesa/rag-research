#!/bin/bash
# CPU-only analysis of a split (labels, PCR, primary/secondary outcomes, gate,
# replication check, figures, summary). Run from the repo or a pinned worktree:
#   SPLIT=full sbatch experiments/whitebox/e2_bridge_mechanism/slurm/analyze_libra.sbatch
#SBATCH --job-name=dbcp_analyze
#SBATCH --partition=ccstar
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=local_data/logs/e2_bridge_mechanism/%x_%j.out
#SBATCH --error=local_data/logs/e2_bridge_mechanism/%x_%j.err
set -euo pipefail
: "${SPLIT:?}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/third_party/open_deep_research/src" HF_HUB_OFFLINE=1
S=$REPO/experiments/whitebox/e2_bridge_mechanism
echo "[analyze] commit $(git rev-parse HEAD) node $(hostname) split $SPLIT"
$HOME/.conda/envs/whitebox/bin/python -B -m prematurestop.pipelines.bridge_mechanism.cli verify-freeze
$HOME/.conda/envs/whitebox/bin/python -B -m prematurestop.pipelines.bridge_mechanism.cli analyze --split "$SPLIT" \
  --state-dir "$S/state" --results-dir "$S/run_results"
echo "[analyze] done"

#!/bin/bash
# Real-model integration test on Libra before dev: smoke split, MAX_QUESTIONS
# per dataset (default 1), ONE cell (default qwen14b/hotpot), every phase
# including all eight arms with provisional equal lambda (smoke only), then
# analyze. Bug-finding only; never used for any claim.
#   VICTIM=llama8b DATASET=2wiki sbatch .../integration_libra.sbatch
#SBATCH --job-name=dbcp_integ
#SBATCH --partition=ccstar
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --nodelist=ml01,ml02
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
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
export SERVE_ENV_SCRIPT="$REPO/experiments/dynamic_bridge_cue_pilot/slurm/libra_vllm_toolchain.sh"
export SMOKE_PROVISIONAL_LAMBDA=1 MAX_QUESTIONS="${MAX_QUESTIONS:-1}"
V="${VICTIM:-qwen14b}"; DS="${DATASET:-hotpot}"
echo "[integ] commit $(git -C "$REPO" rev-parse HEAD) node $(hostname) cell $V/$DS max_q $MAX_QUESTIONS"
D="$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh"
[[ "${SKIP_DECOMPOSE:-0}" == 1 ]] || bash "$D" decompose smoke   # parallel jobs: only one decomposes; cells wait for its marker
bash "$D" cell smoke "$V" "$DS"
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/open_deep_research/src" HF_HUB_OFFLINE=1
"$PY_AN" -B -m attack.dynamic_bridge_pilot.cli analyze --split smoke --state-dir "$STATE_DIR" --results-dir "$RESULTS_DIR" --victims "$V" --datasets "$DS"
echo "[integ] complete"

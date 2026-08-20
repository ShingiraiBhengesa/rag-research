#!/bin/bash
#SBATCH --job-name=pcr_wga_gptoss20b_multihoprag
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gptoss20b/multihoprag/pcr_wga_gptoss20b_multihoprag_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gptoss20b/multihoprag/pcr_wga_gptoss20b_multihoprag_%j.err

set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox

pip install -q duckduckgo-search
mkdir -p "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/gptoss20b/multihoprag" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gptoss20b/multihoprag" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
if [[ ! -s "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_multihoprag.jsonl" ]]; then
  echo "Missing global eligible manifest: /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_multihoprag.jsonl" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE="PAPER_PCR_FULL_COMPLETION_V1"
export WB_RUN_ID="${SLURM_JOB_ID}_wga_gptoss20b_multihoprag"
export WB_MODEL="openai/gpt-oss-20b"
export WB_MAX_STEPS=40
export N_SAMPLE=450
export DATASET_SRC="multihoprag"
export WB_SAMPLE_MANIFEST="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_multihoprag.jsonl"
export DDG_CACHE_DIR="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
export DDG_SLEEP=1.5
export WB_LOG_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/gptoss20b/multihoprag/final_wga_gptoss20b_multihoprag.jsonl"
export WB_DEVICE_MAP="auto"
export WB_BASELINE_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/baselines/gptoss20b/multihoprag/clean_baseline_gptoss20b_multihoprag.jsonl"
if [[ ! -s "$WB_BASELINE_PATH" ]]; then
  echo "Missing clean baseline: $WB_BASELINE_PATH" >&2
  exit 2
fi

cd "/home/sbhengesa/rag-research/whitebox_attack"
python "wrong_gap_assessment.py"
echo "=== DONE rc=$? $(date) ==="

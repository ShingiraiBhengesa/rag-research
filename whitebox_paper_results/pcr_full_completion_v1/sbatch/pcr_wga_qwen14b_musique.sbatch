#!/bin/bash
#SBATCH --job-name=pcr_wga_qwen14b_musique
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/qwen14b/musique/pcr_wga_qwen14b_musique_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/qwen14b/musique/pcr_wga_qwen14b_musique_%j.err

set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox

pip install -q duckduckgo-search
mkdir -p "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/qwen14b/musique" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/qwen14b/musique" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
if [[ ! -s "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_musique.jsonl" ]]; then
  echo "Missing global eligible manifest: /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_musique.jsonl" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE="PAPER_PCR_FULL_COMPLETION_V1"
export WB_RUN_ID="${SLURM_JOB_ID}_wga_qwen14b_musique"
export WB_MODEL="Qwen/Qwen2.5-14B-Instruct"
export WB_MAX_STEPS=40
export N_SAMPLE=450
export DATASET_SRC="musique"
export WB_SAMPLE_MANIFEST="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_musique.jsonl"
export DDG_CACHE_DIR="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
export DDG_SLEEP=1.5
export WB_LOG_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/qwen14b/musique/final_wga_qwen14b_musique.jsonl"
export WB_BASELINE_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/baselines/qwen14b/musique/clean_baseline_qwen14b_musique.jsonl"
if [[ ! -s "$WB_BASELINE_PATH" ]]; then
  echo "Missing clean baseline: $WB_BASELINE_PATH" >&2
  exit 2
fi

cd "/home/sbhengesa/rag-research/whitebox_attack"
python "wrong_gap_assessment.py"
echo "=== DONE rc=$? $(date) ==="

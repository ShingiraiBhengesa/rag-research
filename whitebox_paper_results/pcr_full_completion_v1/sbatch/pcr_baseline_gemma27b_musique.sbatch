#!/bin/bash
#SBATCH --job-name=pcr_base_gemma27b_musique
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gemma27b/musique/pcr_base_gemma27b_musique_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gemma27b/musique/pcr_base_gemma27b_musique_%j.err

set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox

pip install -q duckduckgo-search
mkdir -p "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/baselines/gemma27b/musique" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gemma27b/musique" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
if [[ ! -s "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/samples/musique_n450.jsonl" ]]; then
  echo "Missing sample manifest: /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/samples/musique_n450.jsonl" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE="PAPER_PCR_FULL_COMPLETION_V1"
export WB_RUN_ID="${SLURM_JOB_ID}_baseline_gemma27b_musique"
export WB_MODEL="google/gemma-3-27b-it"
export WB_MAX_STEPS=40
export N_SAMPLE=450
export DATASET_SRC="musique"
export WB_SAMPLE_MANIFEST="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/samples/musique_n450.jsonl"
export DDG_CACHE_DIR="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
export DDG_SLEEP=1.5
export WB_LOG_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/baselines/gemma27b/musique/clean_baseline_gemma27b_musique.jsonl"

cd "/home/sbhengesa/rag-research/whitebox_attack"
python "run_clean_baseline.py"
echo "=== DONE rc=$? $(date) ==="

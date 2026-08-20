#!/bin/bash
#SBATCH --job-name=pcr_factonly_gptoss20b_musique
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gptoss20b/musique/pcr_factonly_gptoss20b_musique_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gptoss20b/musique/pcr_factonly_gptoss20b_musique_%j.err

set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox

pip install -q duckduckgo-search
mkdir -p "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/gptoss20b/musique" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/gptoss20b/musique" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
if [[ ! -s "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_musique.jsonl" ]]; then
  echo "Missing global eligible manifest: /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_musique.jsonl" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE="PAPER_PCR_FULL_COMPLETION_V1"
export WB_RUN_ID="${SLURM_JOB_ID}_factonly_gptoss20b_musique"
export WB_MODEL="openai/gpt-oss-20b"
export WB_MAX_STEPS=40
export N_SAMPLE=450
export DATASET_SRC="musique"
export WB_SAMPLE_MANIFEST="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests/global_eligible_musique.jsonl"
export DDG_CACHE_DIR="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/ddg_cache"
export DDG_SLEEP=1.5
export WB_LOG_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/gptoss20b/musique/final_wga_factonly_gptoss20b_musique.jsonl"
export WB_DEVICE_MAP="auto"
export WB_WGA_PATH="/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/jsonl/gptoss20b/musique/final_wga_gptoss20b_musique.jsonl"
if [[ ! -s "$WB_WGA_PATH" ]]; then
  echo "Missing WGA manifest: $WB_WGA_PATH" >&2
  echo "Submit and complete the matching pcr_wga job first." >&2
  exit 2
fi

cd "/home/sbhengesa/rag-research/whitebox_attack"
python "wrong_gap_assessment_factonly.py"
echo "=== DONE rc=$? $(date) ==="

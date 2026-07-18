#!/bin/bash
#SBATCH --job-name=fo_gemma27b_r
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_attack/logs/fo_gemma27b_rerun_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_attack/logs/fo_gemma27b_rerun_%j.err
# Targeted rerun: multihoprag only, capped at 15 steps.
# Appends to final_wga_factonly_gemma27b.jsonl — 37 already-done examples get duplicates,
# deduplicate on (dataset,example_id,condition) at analysis time.
set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
pip install -q duckduckgo-search
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE=S1
export WB_RUN_ID=$SLURM_JOB_ID
export WB_MODEL=google/gemma-3-27b-it
export WB_MAX_STEPS=15
export N_SAMPLE=450
export DATASET_SRC=multihoprag
export DDG_CACHE_DIR=/home/sbhengesa/rag-research/whitebox_results/ddg_cache
export DDG_SLEEP=1.5
cd /home/sbhengesa/rag-research/whitebox_attack
WB_LOG_PATH=/home/sbhengesa/rag-research/whitebox_results/final_wga_factonly_gemma27b.jsonl \
    python wrong_gap_assessment_factonly.py
echo "=== DONE rc=$? $(date) ==="

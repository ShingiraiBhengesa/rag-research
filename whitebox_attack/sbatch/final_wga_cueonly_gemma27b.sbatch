#!/bin/bash
#SBATCH --job-name=wga_co_gemma27b
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_attack/logs/wga_co_gemma27b_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_attack/logs/wga_co_gemma27b_%j.err
# Submit only after rerun_wga_gemma27b completes (full 4-dataset manifest required).
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
export DATASET_SRC=all
export DDG_CACHE_DIR=/home/sbhengesa/rag-research/whitebox_results/ddg_cache
export DDG_SLEEP=1.5
export WB_WGA_PATH=/home/sbhengesa/rag-research/whitebox_results/final_wga_gemma27b.jsonl
cd /home/sbhengesa/rag-research/whitebox_attack
WB_LOG_PATH=/home/sbhengesa/rag-research/whitebox_results/final_wga_cueonly_gemma27b.jsonl \
    python wrong_gap_assessment_cue_only.py
echo "=== DONE rc=$? $(date) ==="

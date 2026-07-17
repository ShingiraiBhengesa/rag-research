#!/bin/bash
#SBATCH --job-name=wga_fo_llama8b
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_attack/logs/wga_fo_llama8b_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_attack/logs/wga_fo_llama8b_%j.err
set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
pip install -q duckduckgo-search
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE=S1
export WB_RUN_ID=$SLURM_JOB_ID
export WB_MODEL=meta-llama/Llama-3.1-8B-Instruct
export WB_MAX_STEPS=40
export N_SAMPLE=450
export DATASET_SRC=all
export DDG_CACHE_DIR=/home/sbhengesa/rag-research/whitebox_results/ddg_cache
export DDG_SLEEP=1.5
cd /home/sbhengesa/rag-research/whitebox_attack
WB_LOG_PATH=/home/sbhengesa/rag-research/whitebox_results/final_wga_factonly_llama8b.jsonl \
    python wrong_gap_assessment_factonly.py
echo "=== DONE rc=$? $(date) ==="

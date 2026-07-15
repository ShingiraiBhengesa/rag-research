#!/bin/bash
#SBATCH --job-name=s1_a2e_q14b
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_attack/logs/s1_a2e_q14b_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_attack/logs/s1_a2e_q14b_%j.err
set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
pip install -q duckduckgo-search
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE=S1
export WB_RUN_ID=$SLURM_JOB_ID
export WB_MODEL=Qwen/Qwen2.5-14B-Instruct
export N_SAMPLE=450
export DATASET_SRC=all
export DDG_CACHE_DIR=/home/sbhengesa/rag-research/whitebox_results/ddg_cache
export DDG_SLEEP=1.5
cd /home/sbhengesa/rag-research/whitebox_attack
WB_LOG_PATH=/home/sbhengesa/rag-research/whitebox_results/s1_attack2e_pcr10_qwen14b.jsonl python attack2e_delayed_pcr.py
echo "=== DONE rc=$? $(date) ==="

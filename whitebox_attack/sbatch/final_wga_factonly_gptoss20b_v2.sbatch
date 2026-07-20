#!/bin/bash
#SBATCH --job-name=wga_fo2_gptoss20b
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_attack/logs/wga_fo2_gptoss20b_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_attack/logs/wga_fo2_gptoss20b_%j.err
# Manifest-driven FO re-run for GPT-OSS-20B.
# Uses the canonical WGA manifest so eligible set is identical to WGA/CueOnly.
# No clean rollout is run — clean baseline comes from WB_WGA_PATH.
set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
pip install -q duckduckgo-search
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE=S1
export WB_RUN_ID=$SLURM_JOB_ID
export WB_MODEL=openai/gpt-oss-20b
export WB_DEVICE_MAP=auto
export WB_MAX_STEPS=40
export N_SAMPLE=450
export DATASET_SRC=all
export DDG_CACHE_DIR=/home/sbhengesa/rag-research/whitebox_results/ddg_cache
export DDG_SLEEP=1.5
export WB_WGA_PATH=/home/sbhengesa/rag-research/whitebox_results/final_wga_gptoss20b.jsonl
cd /home/sbhengesa/rag-research/whitebox_attack
WB_LOG_PATH=/home/sbhengesa/rag-research/whitebox_results/final_wga_factonly_gptoss20b_v2.jsonl \
    python wrong_gap_assessment_factonly.py
echo "=== DONE rc=$? $(date) ==="

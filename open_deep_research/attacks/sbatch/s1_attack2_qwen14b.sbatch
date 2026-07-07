#!/bin/bash
#SBATCH --job-name=s1_attack2_q14b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=27:00:00
#SBATCH --output=/home/sbhengesa/rag-research/open_deep_research/logs/s1_attack2_q14b_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/open_deep_research/logs/s1_attack2_q14b_%j.err
set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE=S1_retrieval_control_scale
export WB_RUN_ID=$SLURM_JOB_ID
export WB_MODEL=Qwen/Qwen2.5-14B-Instruct
export WB_LOG_PATH=/home/sbhengesa/rag-research/open_deep_research/attacks/results/s1_attack2_gap_assessment_qwen14b.jsonl
export N_SAMPLE=320
export DATASET_SRC=all
cd /home/sbhengesa/rag-research/open_deep_research/attacks
python attack2_gap_assessment.py
echo "=== DONE rc=$? $(date) ==="

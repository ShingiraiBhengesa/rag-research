#!/bin/bash
#SBATCH --job-name=pcr_prepare_samples
#SBATCH --partition=machinelearning
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/_pipeline/pcr_prepare_samples_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/_pipeline/pcr_prepare_samples_%j.err

set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox

mkdir -p "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/_pipeline" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/samples"
cd "/home/sbhengesa/rag-research/whitebox_attack"
python "generate_sample_manifest.py" --out-dir "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/samples" --n-sample 450
echo "=== DONE rc=$? $(date) ==="

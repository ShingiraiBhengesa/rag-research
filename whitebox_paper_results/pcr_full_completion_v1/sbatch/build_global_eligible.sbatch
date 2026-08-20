#!/bin/bash
#SBATCH --job-name=pcr_build_global_eligible
#SBATCH --partition=machinelearning
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/_pipeline/pcr_build_global_eligible_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/_pipeline/pcr_build_global_eligible_%j.err

set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox

mkdir -p "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/logs/_pipeline" "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/manifests"
cd "/home/sbhengesa/rag-research/whitebox_attack"
python "build_global_eligible_manifest.py" --results-root "/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1" --n-sample 450
echo "=== DONE rc=$? $(date) ==="

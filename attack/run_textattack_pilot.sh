#!/bin/bash
#SBATCH --job-name=ta_pilot
#SBATCH --output=/home/sbhengesa/adaptive-rag/logs/ta_pilot_%j.out
#SBATCH --error=/home/sbhengesa/adaptive-rag/logs/ta_pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

cd /home/sbhengesa/adaptive-rag/attack
python run_textattack_pilot.py

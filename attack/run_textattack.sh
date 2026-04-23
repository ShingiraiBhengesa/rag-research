#!/bin/bash
#SBATCH --job-name=textattack_musique
#SBATCH --output=/home/sbhengesa/adaptive-rag/logs/textattack_%j.out
#SBATCH --error=/home/sbhengesa/adaptive-rag/logs/textattack_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

cd ~/adaptive-rag/attack
python run_textattack.py

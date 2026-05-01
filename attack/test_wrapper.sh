#!/bin/bash
#SBATCH --job-name=test_wrapper
#SBATCH --output=/home/sbhengesa/adaptive-rag/logs/test_wrapper_%j.out
#SBATCH --error=/home/sbhengesa/adaptive-rag/logs/test_wrapper_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

cd ~/adaptive-rag/attack
python classifier_wrapper.py

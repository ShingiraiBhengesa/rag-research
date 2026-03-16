#!/bin/bash
#SBATCH --job-name=selfrag_popqa
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate selfrag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/selfrag/lib:$LD_LIBRARY_PATH

cd ~/selfrag/repos/self-rag/retrieval_lm

echo "=========================================="
echo "MODE 1: ADAPTIVE RETRIEVAL (Self-RAG decides when to retrieve)"
echo "=========================================="
python run_short_form.py \
  --model_name /home/sbhengesa/selfrag/models/selfrag_llama2_7b \
  --input_file eval_data/popqa_longtail_w_gs.jsonl \
  --mode adaptive_retrieval \
  --max_new_tokens 100 \
  --threshold 0.2 \
  --output_file ~/selfrag/results/popqa_adaptive.json \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
  --dtype half

echo "=========================================="
echo "MODE 2: ALWAYS RETRIEVE"
echo "=========================================="
python run_short_form.py \
  --model_name /home/sbhengesa/selfrag/models/selfrag_llama2_7b \
  --input_file eval_data/popqa_longtail_w_gs.jsonl \
  --mode always_retrieve \
  --max_new_tokens 100 \
  --threshold 0.2 \
  --output_file ~/selfrag/results/popqa_always_retrieve.json \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
  --dtype half

echo "=========================================="
echo "MODE 3: NO RETRIEVAL (pure parametric memory)"
echo "=========================================="
python run_short_form.py \
  --model_name /home/sbhengesa/selfrag/models/selfrag_llama2_7b \
  --input_file eval_data/popqa_longtail_w_gs.jsonl \
  --mode no_retrieval \
  --max_new_tokens 100 \
  --threshold 0.2 \
  --output_file ~/selfrag/results/popqa_no_retrieval.json \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
  --dtype half

echo "DONE! All three modes completed."

#!/bin/bash
#SBATCH --job-name=selfrag_triviaqa
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=selfrag_triviaqa_%j.out
#SBATCH --error=selfrag_triviaqa_%j.err

module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate selfrag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/selfrag/lib:$LD_LIBRARY_PATH

cd ~/selfrag/repos/self-rag/retrieval_lm

echo "MODE 1: ADAPTIVE RETRIEVAL"
python run_short_form.py \
  --model_name /home/sbhengesa/selfrag/models/selfrag_llama2_7b \
  --input_file eval_data/triviaqa_test_w_gs.jsonl \
  --mode adaptive_retrieval --max_new_tokens 100 --threshold 0.2 \
  --output_file ~/selfrag/results/triviaqa_adaptive.json \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore --dtype half

echo "MODE 2: ALWAYS RETRIEVE"
python run_short_form.py \
  --model_name /home/sbhengesa/selfrag/models/selfrag_llama2_7b \
  --input_file eval_data/triviaqa_test_w_gs.jsonl \
  --mode always_retrieve --max_new_tokens 100 --threshold 0.2 \
  --output_file ~/selfrag/results/triviaqa_always_retrieve.json \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore --dtype half

echo "MODE 3: NO RETRIEVAL"
python run_short_form.py \
  --model_name /home/sbhengesa/selfrag/models/selfrag_llama2_7b \
  --input_file eval_data/triviaqa_test_w_gs.jsonl \
  --mode no_retrieval --max_new_tokens 100 --threshold 0.2 \
  --output_file ~/selfrag/results/triviaqa_no_retrieval.json \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore --dtype half

echo "DONE!"

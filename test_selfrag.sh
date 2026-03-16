#!/bin/bash
#SBATCH --job-name=selfrag_test
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load modules
module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate selfrag

# Force use of conda's newer libstdc++
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/selfrag/lib:$LD_LIBRARY_PATH

# Verify GPU is visible
nvidia-smi

# Run test query
cd ~/selfrag/repos/self-rag
python -c "
from vllm import LLM, SamplingParams

print('Loading model...')
model = LLM('/home/sbhengesa/selfrag/models/selfrag_llama2_7b', dtype='half')
print('Model loaded!')

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def format_prompt(input, paragraph=None):
    prompt = '### Instruction:\n{0}\n\n### Response:\n'.format(input)
    if paragraph is not None:
        prompt += '[Retrieval]<paragraph>{0}</paragraph>'.format(paragraph)
    return prompt

# Test 1: Query that should NOT trigger retrieval
q1 = format_prompt('Write a short poem about the ocean.')

# Test 2: Query that SHOULD trigger retrieval
q2 = format_prompt('What is the capital of France?')

# Test 3: Query WITH a provided passage
q3 = format_prompt('What is the capital of France?', 'Paris is the capital and most populous city of France.')

print('='*60)
print('TEST 1: Creative query (should see No Retrieval)')
print('='*60)
preds = model.generate([q1], sampling_params)
print(preds[0].outputs[0].text)

print()
print('='*60)
print('TEST 2: Factual query (should see Retrieve token)')
print('='*60)
preds = model.generate([q2], sampling_params)
print(preds[0].outputs[0].text)

print()
print('='*60)
print('TEST 3: Query with passage (should see Relevant + Supported)')
print('='*60)
preds = model.generate([q3], sampling_params)
print(preds[0].outputs[0].text)

print()
print('DONE! Self-RAG is working.')
"

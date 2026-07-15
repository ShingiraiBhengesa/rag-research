#!/bin/bash
#SBATCH --job-name=s1_a2e_gptoss120b
#SBATCH --partition=machinelearning
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=40:00:00
#SBATCH --output=/home/sbhengesa/rag-research/whitebox_attack/logs/s1_a2e_gptoss120b_%j.out
#SBATCH --error=/home/sbhengesa/rag-research/whitebox_attack/logs/s1_a2e_gptoss120b_%j.err
set -uo pipefail
echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
pip install -q duckduckgo-search "kernels==0.14.1"

# Triton compiles cuda_utils.c at runtime; spack GCC 14.2.0 on this cluster is
# missing crtbeginS.o in its linker search path. Wrap gcc to inject -B so ld finds it.
_SPACK_GCC=/cm/shared/apps/spack/opt/spack/linux-rhel9-sapphirerapids/gcc-13.1.0/gcc-14.2.0-7qanuh5jgy3lgxvl67v3qkva2mya7c2a
_GCC_RTDIR="${_SPACK_GCC}/lib/gcc/x86_64-pc-linux-gnu/14.2.0"
_WRAP_DIR="/tmp/gcc_wrap_${SLURM_JOB_ID}"
mkdir -p "${_WRAP_DIR}"
printf '#!/bin/bash\nexec "%s/bin/gcc" -B"%s" "$@"\n' "${_SPACK_GCC}" "${_GCC_RTDIR}" > "${_WRAP_DIR}/gcc"
chmod +x "${_WRAP_DIR}/gcc"
export PATH="${_WRAP_DIR}:${PATH}"
# Triton checks CC env var first; set it explicitly so it uses the wrapper
export CC="${_WRAP_DIR}/gcc"
echo "=== GCC wrapper: $(${_WRAP_DIR}/gcc --version 2>&1 | head -1) ==="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WB_STAGE=S1
export WB_RUN_ID=$SLURM_JOB_ID
export WB_MODEL=openai/gpt-oss-120b
export WB_DEVICE_MAP=auto
export N_SAMPLE=450
export DATASET_SRC=all
export DDG_CACHE_DIR=/home/sbhengesa/rag-research/whitebox_results/ddg_cache
export DDG_SLEEP=1.5
cd /home/sbhengesa/rag-research/whitebox_attack
WB_LOG_PATH=/home/sbhengesa/rag-research/whitebox_results/s1_attack2e_pcr10_gptoss120b.jsonl python attack2e_delayed_pcr.py
echo "=== DONE rc=$? $(date) ==="

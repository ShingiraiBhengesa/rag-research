# Sourced ONLY around the vLLM server process on Libra (run_phase.sh SERVE_ENV_SCRIPT).
# Copied from the canonical production sbatch (results/sbatch/run_v3_*.sbatch):
# Triton/xgrammar JIT-compile CUDA helpers on the first request; the vllm conda
# env's cross-compiler is broken and the loaded Spack GCC cannot find its runtime
# objects (crtbeginS.o, libgcc) on these nodes, which killed the engine in job
# 173943. Pin the complete RHEL system GCC toolchain, exactly as canonical does.
CONDA_PREFIX="${PREMATURESTOP_VLLM_ENV:-$HOME/.conda/envs/vllm}"
SYSTEM_GCC=/usr/bin/gcc
SYSTEM_GXX=/usr/bin/g++
SYSTEM_GCC_VERSION=11
SYSTEM_GCC_TARGET=x86_64-redhat-linux
SYSTEM_GCC_LIB=/usr/lib/gcc/$SYSTEM_GCC_TARGET/$SYSTEM_GCC_VERSION
SYSTEM_GCC_LIBEXEC=/usr/libexec/gcc/$SYSTEM_GCC_TARGET/$SYSTEM_GCC_VERSION
if [[ ! -x "$SYSTEM_GCC" || ! -x "$SYSTEM_GXX" || ! -x "$SYSTEM_GCC_LIBEXEC/cc1" || ! -f "$SYSTEM_GCC_LIB/crtbeginS.o" || ! -f "$SYSTEM_GCC_LIB/libgcc.a" ]]; then
  echo "System GCC toolchain is incomplete: lib=$SYSTEM_GCC_LIB libexec=$SYSTEM_GCC_LIBEXEC" >&2
  exit 3
fi
export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
unset GCC_EXEC_PREFIX CPATH
export CUDA_HOME=/cm/local/apps/cuda/libs/current
export CC="$SYSTEM_GCC" CXX="$SYSTEM_GXX" GCC="$SYSTEM_GCC" LD=/usr/bin/ld AS=/usr/bin/as AR=/usr/bin/ar
export COMPILER_PATH="$SYSTEM_GCC_LIBEXEC:$SYSTEM_GCC_LIB"
export LIBRARY_PATH="$SYSTEM_GCC_LIB:/usr/lib64:/usr/lib"
export C_INCLUDE_PATH="$SYSTEM_GCC_LIB/include:/usr/include"
export CPLUS_INCLUDE_PATH="$SYSTEM_GCC_LIB/include:/usr/include/c++/$SYSTEM_GCC_VERSION:/usr/include/c++/$SYSTEM_GCC_VERSION/$SYSTEM_GCC_TARGET:/usr/include"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="${STATE_DIR:-/tmp}/triton_cache/${SLURM_JOB_ID:-local}"
mkdir -p "$TRITON_CACHE_DIR"
# Same compiler smoke test as canonical: fail fast instead of on the first request.
_smoke="$(mktemp -d /tmp/dbcp_compiler_smoke.XXXXXX)"
printf '#include <Python.h>\nint dbcp_smoke(void){return 0;}\n' > "$_smoke/smoke.c"
"$CC" "$_smoke/smoke.c" -O3 -shared -fPIC -Wno-psabi -o "$_smoke/smoke.so" -l:libcuda.so.1 \
  -L"$CONDA_PREFIX/lib/python3.11/site-packages/triton/backends/nvidia/lib" -L"$CUDA_HOME/lib64" \
  -I"$CONDA_PREFIX/include/python3.11" -I"$CONDA_PREFIX/lib/python3.11/site-packages/triton/backends/nvidia/include" \
  || { echo "compiler smoke test failed" >&2; rm -rf "$_smoke"; exit 3; }
rm -rf "$_smoke"

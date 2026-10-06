#!/bin/bash
# Preflight checks to run INSIDE the NRP pod before a pilot launch.
# Prints what it finds; does not invent values. Fails loudly on the essentials.
set -uo pipefail

echo "== host =="; hostname; echo
echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 || echo "nvidia-smi unavailable"; echo
echo "== CPU/RAM =="; nproc; free -h 2>&1 | head -2; echo
echo "== storage mounts (expect a persistent writable PVC for results) =="
df -h 2>&1 | grep -Ev "tmpfs|overlay" || true
echo "writable test on \$PILOT_PVC=${PILOT_PVC:-UNSET}:"
if [[ -n "${PILOT_PVC:-}" ]]; then
  touch "$PILOT_PVC/.pilot_write_test" && echo "  writable OK" && rm -f "$PILOT_PVC/.pilot_write_test" || echo "  NOT writable"
else
  echo "  PILOT_PVC not set -- set it to the mounted PVC path"
fi
echo
echo "== conda envs (need odr for the runner, vllm for serving) =="
ls "${CONDA_ROOT:-/opt/conda}/envs" 2>&1 || ls ~/.conda/envs 2>&1 || echo "no conda envs found"
echo
echo "== vLLM endpoints =="
for url in "${VICTIM_URL:-}" "${ATTACKER_URL:-}"; do
  [[ -z "$url" ]] && continue
  echo "  probing $url/models"
  python -c "import urllib.request,sys; print('  OK', urllib.request.urlopen('$url/models', timeout=5).status)" 2>&1 || echo "  UNREACHABLE"
done
echo
echo "== HF dataset/model cache (for offline split build and serving) =="
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub" 2>&1 | tail -1 || echo "no hub cache"
echo "preflight done"

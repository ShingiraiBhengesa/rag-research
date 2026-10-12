#!/usr/bin/env python3
"""Render the NRP Job manifests for one split at one frozen commit.

    python render_jobs.py --split smoke --commit <sha> [--namespace trace-ai]

    python render_jobs.py --split full --commit <sha> --shards 8

Writes nrp/k8s/rendered/<split>/{00-setup,01-decompose,02-cell-*}.yaml.
With --shards N each cell is an Indexed Job of N pods (1 L40 each); every
pod takes a fixed 1/N slice of the questions and checkpoints to its own files.
Apply in order: setup (once per commit), decompose, then the four cells
(the cells wait for the decompose marker on the PVC by themselves).

Infrastructure from the NRP brief: priorityClassName opportunistic (pods can be
preempted -> every phase checkpoints per unit to the PVC), NVIDIA-L40 nodes,
1 GPU / 48Gi / 8 CPU per job, PVC adaptive-pilot-work mounted at /pvc
(RWX; the four cells write disjoint files). The V100 fallback with 4-bit
quantisation is deliberately NOT rendered: see ../README.md.
"""
import argparse
from pathlib import Path

VICTIMS = ("qwen14b", "llama8b")
DATASETS = ("hotpot", "2wiki")

POD_COMMON = """\
      priorityClassName: opportunistic
      restartPolicy: OnFailure
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-L40
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: adaptive-pilot-work
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
"""

ENV = """\
          env:
            - {{name: REPO, value: /pvc/code/{commit}}}
            - {{name: PY_VLLM, value: python3}}
            - {{name: PY_ODR, value: /pvc/envs/odr-{commit}/bin/python}}
            - {{name: HF_HOME, value: /pvc/hf}}
            - {{name: STATE_DIR, value: /pvc/checkpoints/dynamic_bridge_cue_pilot}}
            - {{name: RESULTS_DIR, value: /pvc/results/dynamic_bridge_cue_pilot}}
            - {{name: SMOKE_PROVISIONAL_LAMBDA, value: "{smoke_lambda}"}}
"""

RESOURCES = """\
          resources:
            requests: {{nvidia.com/gpu: "{gpu}", memory: "48Gi", cpu: "8"}}
            limits: {{nvidia.com/gpu: "{gpu}", memory: "48Gi", cpu: "8"}}
          volumeMounts:
            - {{name: work, mountPath: /pvc}}
            - {{name: dshm, mountPath: /dev/shm}}
"""


def job(name, namespace, image, args, *, gpu, commit, smoke_lambda, extra_env="", shards=1, stop_after=""):
    indexed = (f"  completionMode: Indexed\n  completions: {shards}\n  parallelism: {shards}\n"
               if shards > 1 else "")
    if shards > 1:
        extra_env += f"            - {{name: NUM_SHARDS, value: \"{shards}\"}}\n"
    if stop_after:
        extra_env += f"            - {{name: STOP_AFTER, value: \"{stop_after}\"}}\n"
    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {namespace}
  labels: {{app: dynamic-bridge-cue-pilot, commit: "{commit[:12]}"}}
spec:
{indexed}  backoffLimit: {50 * shards}
  template:
    metadata:
      labels: {{app: dynamic-bridge-cue-pilot}}
    spec:
{POD_COMMON}      containers:
        - name: main
          image: {image}
          command: ["/bin/bash", "-lc"]
          args:
            - |
{chr(10).join('              ' + line for line in args.strip().splitlines())}
{ENV.format(commit=commit, smoke_lambda=smoke_lambda)}{extra_env}{RESOURCES.format(gpu=gpu)}"""


SETUP = """\
set -euo pipefail
# 1) code at the frozen commit (read-only thereafter)
if [ ! -d "$REPO/.git" ]; then
  git clone --no-checkout "https://x-access-token:${GITHUB_TOKEN}@github.com/ShingiraiBhengesa/rag-research.git" "$REPO"
  git -C "$REPO" checkout --detach {commit}
fi
test "$(git -C "$REPO" rev-parse HEAD)" = "{commit}"
git -C "$REPO" remote set-url origin https://github.com/ShingiraiBhengesa/rag-research.git
# 2) driver env layered on the serving image (torch/transformers/vllm from the image)
if [ ! -x "$PY_ODR" ]; then
  python3 -m venv --system-site-packages "$(dirname "$(dirname "$PY_ODR")")"
  "$PY_ODR" -m pip install -q -e "$REPO/open_deep_research" -r "$REPO/requirements-whitebox.txt"
fi
# 3) weights at the pinned revisions + the pinned NLI checker
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev in [("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"),
                  ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
                  ("cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7")]:
    print(snapshot_download(repo, revision=rev))
PY
# 4) offline tests inside the pod environment (same code as validated on Libra)
cd "$REPO" && PYTHONPATH="$REPO:$REPO/open_deep_research/src" HF_HUB_OFFLINE=1 \\
  "$PY_ODR" -m unittest attack.dynamic_bridge_pilot.tests.test_pilot
echo SETUP_OK
"""

SECRETS = """\
            - name: GITHUB_TOKEN
              valueFrom: {secretKeyRef: {name: github-token, key: token}}
            - name: HF_TOKEN
              valueFrom: {secretKeyRef: {name: hf-token, key: token}}
"""


TOOLBOX = """\
# No-GPU pod with the PVC and the pilot env, for audit-export, analyze and
# copying files in/out (kubectl cp / exec). Delete it when done.
apiVersion: v1
kind: Pod
metadata:
  name: dbcp-toolbox-{short}
  namespace: {namespace}
  labels: {{app: dynamic-bridge-cue-pilot}}
spec:
  priorityClassName: opportunistic
  restartPolicy: Never
  containers:
    - name: main
      image: {image}
      command: ["/bin/bash", "-lc", "sleep infinity"]
      env:
        - {{name: REPO, value: /pvc/code/{commit}}}
        - {{name: PY_ODR, value: /pvc/envs/odr-{commit}/bin/python}}
        - {{name: PYTHONPATH, value: /pvc/code/{commit}:/pvc/code/{commit}/open_deep_research/src}}
        - {{name: HF_HOME, value: /pvc/hf}}
        - {{name: HF_HUB_OFFLINE, value: "1"}}
        - {{name: STATE_DIR, value: /pvc/checkpoints/dynamic_bridge_cue_pilot}}
        - {{name: RESULTS_DIR, value: /pvc/results/dynamic_bridge_cue_pilot}}
      resources:
        requests: {{memory: "16Gi", cpu: "4"}}
        limits: {{memory: "16Gi", cpu: "4"}}
      volumeMounts:
        - {{name: work, mountPath: /pvc}}
  volumes:
    - name: work
      persistentVolumeClaim:
        claimName: adaptive-pilot-work
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("smoke", "dev", "full"))
    ap.add_argument("--commit", required=True)
    ap.add_argument("--namespace", default="trace-ai")
    ap.add_argument("--image", default="vllm/vllm-openai:v0.23.0")
    ap.add_argument("--shards", type=int, default=1,
                    help="pods per victim x dataset cell (Indexed Job). Shard i runs every N-th question; "
                         "results are identical to N=1, only wall time changes. Suggested: 1 for dev, 8 for full.")
    ap.add_argument("--stop-after", choices=("", "anchor"), default="",
                    help="anchor = cells run only the clean anchor runs (baseline gate before any cue/arm work)")
    a = ap.parse_args()
    if len(a.commit) != 40:
        ap.error("--commit must be the full 40-char SHA of the pushed, frozen branch head")
    out = Path(__file__).resolve().parent / "rendered" / a.split
    out.mkdir(parents=True, exist_ok=True)
    smoke_lambda = "1" if a.split == "smoke" else "0"
    short = a.commit[:8]
    common = dict(commit=a.commit, smoke_lambda=smoke_lambda)
    (out / "00-setup.yaml").write_text(job(f"dbcp-setup-{short}", a.namespace, a.image, SETUP.replace("{commit}", a.commit),
                                           gpu=0, extra_env=SECRETS, **common))
    (out / "01-decompose.yaml").write_text(job(
        f"dbcp-{a.split}-decompose-{short}", a.namespace, a.image,
        f'bash "$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh" decompose {a.split}', gpu=1, **common))
    for v in VICTIMS:
        for d in DATASETS:
            (out / f"02-cell-{v}-{d}.yaml").write_text(job(
                f"dbcp-{a.split}-{v}-{d}-{short}", a.namespace, a.image,
                f'bash "$REPO/experiments/dynamic_bridge_cue_pilot/nrp/run_phase.sh" cell {a.split} {v} {d}',
                gpu=1, shards=a.shards, stop_after=a.stop_after, **common))
    (out / "03-toolbox.yaml").write_text(TOOLBOX.format(namespace=a.namespace, image=a.image, commit=a.commit,
                                                        short=short))
    print(f"wrote {out} ({a.shards} shard(s) per cell, {4 * a.shards} cell pods)")


if __name__ == "__main__":
    main()

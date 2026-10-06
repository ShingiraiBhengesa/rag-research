# NRP execution assets for the adaptive-cue pilot

> **Blocker recorded, not worked around.** These assets were written on the
> Libra login node (`login01`, SLURM, `ccstar` partition). This node has **no
> `kubectl`, no kubeconfig, and no network path to the NRP/Nautilus cluster or
> the `trace-ai` namespace**. The pod environment therefore could **not** be
> inspected from here: GPU type/count, PVC names, image, service endpoints, and
> capacity are **unverified**. Every placeholder below marked `<<...>>` must be
> filled in and checked from inside an NRP pod before launching. Nothing here
> invents a namespace, PVC, GPU type, image, or credential as fact.

## What the pilot needs from NRP

1. Two OpenAI-compatible vLLM endpoints, or one shared endpoint:
   - **Victim**: one of `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`,
     `meta-llama/Llama-3.1-8B-Instruct` at the pinned revisions in
     `attack/adaptive_pilot/config.py`.
   - **Attacker**: `Qwen/Qwen2.5-14B-Instruct` (fixed across victims).
   - Sharing one served `Qwen2.5-14B-Instruct` process for the qwen14b victim
     and the attacker is acceptable: victim runs are isolated by `run_id`/thread
     and `MemorySaver`, attacker calls are stateless. Measure contention
     (see `measure_throughput.py`) before sharing.
2. Victim serving flags identical to canonical production
   (`attack/launch/generate_production_arrays.py`): `--dtype bfloat16`,
   `--generation-config vllm`, `--enable-auto-tool-choice`, tool-call parser
   `hermes` (qwen) / `llama3_json` (llama), `--max-model-len 32768`,
   `--enforce-eager`. **Do not change precision or quantization** to fit memory;
   if it will not fit, stage execution instead (see below).
3. Persistent writable storage (an NRP PVC) for `results/adaptive_pilot/`
   (traces, cells, splits, freeze manifest, logs). Do **not** rely on ephemeral
   pod storage.
4. Network access from the pod to the vLLM endpoint(s) and to the local corpus
   files on the PVC. No external retrieval service is used (retrieval is
   deterministic local BM25).

## Staged execution (if attacker + victim do not fit concurrently)

The pilot runner already separates the steps; stage them per
`attack/adaptive_pilot/config.py` and the task spec:

1. Generate Dynamic suffixes (attacker endpoint up; victim down).
2. Run victim first attempts (Control/Static/Dynamic).
3. Generate revised suffixes from saved whitelisted feedback (attacker up).
4. Run revised victim attempts (Optimized).

Each stage reads/writes the same append-only `*_cells.jsonl`; `cell_id`
idempotence makes reruns safe. Per-question state isolation is preserved
because every arm has its own `run_id`.

## Files here

- `vllm_serve.sh` — parameterized vLLM launch (victim or attacker). Placeholders
  must be verified against the pod.
- `pod_preflight.sh` — checks GPU, memory, storage mount, and endpoint before a run.
- `measure_throughput.py` — bounded-concurrency throughput/latency probe for one
  endpoint, to measure contention before scaling.
- `pilot_pod.example.yaml` — a **template** Kubernetes Job/Pod spec. Namespace,
  PVC, image, GPU resource keys, and node selectors are placeholders.

## Launch commands (after placeholders are verified)

```bash
# 0. Build splits on a node with the HF dataset cache (offline):
HF_HUB_OFFLINE=1 python -B -m attack.adaptive_pilot.split_manifest \
  --out-dir /pvc/results/adaptive_pilot/splits

# 1. Smoke test (1 question/dataset x 3 victims x 1 cue x 4 arms = 24 victim runs):
python -B -m attack.adaptive_pilot.cli --split smoke \
  --victim-base-url  "$VICTIM_URL"  --attacker-base-url "$ATTACKER_URL" \
  --conditions Natural_source_coverage --max-questions-per-dataset 1 \
  --split-dir /pvc/results/adaptive_pilot/splits \
  --out-root /pvc/results/adaptive_pilot/smoke

# 2. Development (20 questions x 3 victims x 3 cues x arms):
python -B -m attack.adaptive_pilot.cli --split dev \
  --victim-base-url "$VICTIM_URL" --attacker-base-url "$ATTACKER_URL" \
  --split-dir /pvc/results/adaptive_pilot/splits \
  --out-root /pvc/results/adaptive_pilot/dev

# 3. Freeze before held-out:
python -B -m attack.adaptive_pilot.freeze \
  --split-dir /pvc/results/adaptive_pilot/splits \
  --out /pvc/results/adaptive_pilot/FROZEN_PROCEDURE.json

# 4. Held-out (40 questions):
python -B -m attack.adaptive_pilot.cli --split heldout \
  --victim-base-url "$VICTIM_URL" --attacker-base-url "$ATTACKER_URL" \
  --split-dir /pvc/results/adaptive_pilot/splits \
  --out-root /pvc/results/adaptive_pilot/heldout
```

The pilot `cli.py` runner runs in the `odr` conda env (langgraph 1.2.6), the
same env canonical production uses; the vLLM server runs in the `vllm` env.
Both must be present in the pod image.

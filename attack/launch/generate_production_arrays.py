#!/usr/bin/env python3
"""Generate, validate, and submit the frozen final-production arrays."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from attack.runtime.cues import (
    CONTINUE_SEARCH_SUFFIX,
    DEPRECATED_CONDITION_ALIASES,
    NEUTRAL_SUFFIX,
    PREREGISTERED_SUFFICIENCY_SATURATION_SUFFIX,
    PREREGISTERED_SUFFICIENCY_SUFFIX,
    V3_CONDITIONS,
    suffix_for_condition,
)
from attack.runtime.hashing import hash_text, stable_hash
from attack.runtime.production_manifest import (
    DEFAULT_PRODUCTION_ROOT,
    MANIFEST_DATASET_NAMES,
    N_SAMPLE,
    DATASETS,
    freeze_all_manifests,
    multihoprag_oracle_audit,
)
from attack.runtime.production_models import (
    FINAL_MODEL_IDS,
    FINAL_PRODUCTION_MODELS,
    MODEL_REASONING_PARSERS,
    MODEL_TOOL_CALL_PARSERS,
    PIPELINE_VALIDATION_ONLY_MODEL_IDS,
)
from attack.runtime.production_runner import stable_run_id


REPO_ROOT = Path(__file__).resolve().parents[2]
WHITEBOX_PYTHON = Path("/home/sbhengesa/.conda/envs/whitebox/bin/python")
ODR_PYTHON = Path("/home/sbhengesa/.conda/envs/odr/bin/python")
VLLM_CONDA_ENV = "vllm"
ODR_CONDA_ENV = "odr"
DEV_BUNDLE = REPO_ROOT / "results" / "nonproduction_validation" / "dev_bundle"
MODEL_IDS = dict(FINAL_MODEL_IDS)
SCRIPT_DATASET_NAMES = {"hotpot": "hotpotqa", "2wiki": "2wiki", "musique": "musique", "multihoprag": "multihoprag"}
SCRIPT_NAMES = {
    (dataset, model): f"run_v3_{SCRIPT_DATASET_NAMES[dataset]}_{model}.sbatch"
    for dataset in DATASETS
    for model in FINAL_PRODUCTION_MODELS
}


@dataclass(frozen=True)
class ModelResources:
    model: str
    partition: str
    gres: str
    nodelist: str
    cpus: int
    mem: str
    wall_time: str
    shards_per_condition: int
    examples_per_shard: int
    initial_throttle: int
    gpu_memory_utilization: str
    max_model_len: int
    tensor_parallel_size: int
    tool_call_parser: str
    tool_parser_plugin: str
    reasoning_parser: str
    research_model_max_tokens: int
    compression_model_max_tokens: int
    final_report_model_max_tokens: int
    vllm_ready_timeout_sec: int
    concurrency_note: str


MODEL_RESOURCES = {
    "qwen14b": ModelResources(
        model="qwen14b",
        partition="ccstar",
        gres="gpu:nvidia_h100_nvl:1",
        nodelist="ml01,ml02",
        cpus=10,
        mem="180G",
        wall_time="24:00:00",
        shards_per_condition=9,
        examples_per_shard=50,
        initial_throttle=1,
        gpu_memory_utilization="0.84",
        max_model_len=32768,
        tensor_parallel_size=1,
        tool_call_parser=MODEL_TOOL_CALL_PARSERS["qwen14b"],
        tool_parser_plugin="",
        reasoning_parser="",
        research_model_max_tokens=1200,
        compression_model_max_tokens=2400,
        final_report_model_max_tokens=1200,
        vllm_ready_timeout_sec=5400,
        concurrency_note="raise to 1-2 active tasks only after gate/runtime inspection",
    ),
    "gemma27b": ModelResources(
        model="gemma27b",
        partition="ccstar",
        gres="gpu:nvidia_h100_nvl:1",
        nodelist="ml01,ml02",
        cpus=12,
        mem="220G",
        wall_time="30:00:00",
        shards_per_condition=9,
        examples_per_shard=50,
        initial_throttle=1,
        gpu_memory_utilization="0.82",
        max_model_len=8192,
        tensor_parallel_size=1,
        tool_call_parser=MODEL_TOOL_CALL_PARSERS["gemma27b"],
        tool_parser_plugin="attack/runtime/gemma3_tool_code_parser.py",
        reasoning_parser="",
        research_model_max_tokens=1200,
        compression_model_max_tokens=2400,
        final_report_model_max_tokens=1200,
        vllm_ready_timeout_sec=7200,
        concurrency_note="keep at one active task until gate/runtime inspection passes",
    ),
    "gpt_oss_20b": ModelResources(
        model="gpt_oss_20b",
        partition="ccstar",
        gres="gpu:nvidia_h100_nvl:1",
        nodelist="ml01,ml02",
        cpus=12,
        mem="220G",
        wall_time="30:00:00",
        shards_per_condition=9,
        examples_per_shard=50,
        initial_throttle=1,
        gpu_memory_utilization="0.82",
        max_model_len=32768,
        tensor_parallel_size=1,
        tool_call_parser=MODEL_TOOL_CALL_PARSERS["gpt_oss_20b"],
        tool_parser_plugin="",
        reasoning_parser=MODEL_REASONING_PARSERS["gpt_oss_20b"],
        research_model_max_tokens=1200,
        compression_model_max_tokens=2400,
        final_report_model_max_tokens=2400,
        vllm_ready_timeout_sec=7200,
        concurrency_note="keep at one active task until gate/runtime inspection passes",
    ),
    "llama8b": ModelResources(
        model="llama8b",
        partition="ccstar",
        gres="gpu:nvidia_h100_nvl:1",
        nodelist="ml01,ml02",
        cpus=10,
        mem="180G",
        wall_time="24:00:00",
        shards_per_condition=9,
        examples_per_shard=50,
        initial_throttle=1,
        gpu_memory_utilization="0.82",
        max_model_len=32768,
        tensor_parallel_size=1,
        tool_call_parser=MODEL_TOOL_CALL_PARSERS["llama8b"],
        tool_parser_plugin="",
        reasoning_parser="",
        research_model_max_tokens=1200,
        compression_model_max_tokens=2400,
        final_report_model_max_tokens=1200,
        vllm_ready_timeout_sec=5400,
        concurrency_note="raise to 1-2 active tasks only after gate/runtime inspection",
    ),
}


def production_conditions(*, multihoprag_oracle_allowed: bool, dataset: str) -> tuple[str, ...]:
    if dataset == "multihoprag" and not multihoprag_oracle_allowed:
        return tuple(condition for condition in V3_CONDITIONS if not condition.startswith("OracleComplete_"))
    return tuple(V3_CONDITIONS)


def generate_artifacts(*, production_root: Path, local_files_only: bool) -> dict:
    if "v1" in str(production_root) or "v2" in str(production_root):
        raise ValueError(f"refusing v1/v2 production root: {production_root}")
    freeze_report = freeze_all_manifests(production_root=production_root, local_files_only=local_files_only)
    oracle_audit = multihoprag_oracle_audit(production_root)
    sbatch_dir = production_root / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = build_expected_run_manifest(production_root=production_root, multihoprag_oracle_allowed=oracle_audit["allowed"])
    scripts = []
    for dataset in DATASETS:
        for model in FINAL_PRODUCTION_MODELS:
            (production_root / "logs" / model / dataset).mkdir(parents=True, exist_ok=True)
            script_path = sbatch_dir / SCRIPT_NAMES[(dataset, model)]
            script_path.write_text(
                render_sbatch(
                    production_root=production_root,
                    dataset=dataset,
                    model=model,
                    conditions=production_conditions(multihoprag_oracle_allowed=oracle_audit["allowed"], dataset=dataset),
                ),
                encoding="utf-8",
            )
            scripts.append(str(script_path))
    wrapper_path = production_root / "submit_v3_final_production_arrays.sh"
    wrapper_path.write_text(render_submit_wrapper(production_root=production_root, scripts=scripts), encoding="utf-8")
    wrapper_path.chmod(0o755)
    matrix = {
        "schema_version": "v3_final_production_matrix_v1",
        "production_root": str(production_root),
        "models": list(FINAL_PRODUCTION_MODELS),
        "model_ids": MODEL_IDS,
        "pipeline_validation_only_models": dict(PIPELINE_VALIDATION_ONLY_MODEL_IDS),
        "datasets": list(DATASETS),
        "conditions": list(V3_CONDITIONS),
        "deprecated_condition_aliases": DEPRECATED_CONDITION_ALIASES,
        "multihoprag_oracle_audit": oracle_audit,
        "scripts": scripts,
        "submit_wrapper": str(wrapper_path),
        "expected_run_manifest": str(run_manifest["path"]),
        "expected_total_runs": run_manifest["n"],
        "expected_total_array_tasks": run_manifest["array_tasks"],
        "manifest_freeze": freeze_report,
        "frozen_pipeline_fingerprint": frozen_pipeline_fingerprint(
            production_root=production_root,
            manifest_reports=freeze_report.get("datasets", []),
        ),
        "model_resources": {model: asdict(resources) for model, resources in MODEL_RESOURCES.items()},
    }
    matrix_path = production_root / "matrix" / "v3_final_production_matrix.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix


def build_expected_run_manifest(*, production_root: Path, multihoprag_oracle_allowed: bool) -> dict:
    rows = []
    for dataset in DATASETS:
        manifest = _manifest_path(production_root, dataset)
        examples = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for model, resources in MODEL_RESOURCES.items():
            conditions = production_conditions(multihoprag_oracle_allowed=multihoprag_oracle_allowed, dataset=dataset)
            for condition_index, condition in enumerate(conditions):
                for sample_index, example in enumerate(examples):
                    shard_index = sample_index % resources.shards_per_condition
                    array_index = condition_index * resources.shards_per_condition + shard_index
                    rows.append(
                        {
                            "schema_version": "v3_expected_run_v1",
                            "run_id": stable_run_id(
                                model=model,
                                dataset=dataset,
                                condition=condition,
                                example_id=example["example_id"],
                            ),
                            "model": model,
                            "model_id": MODEL_IDS[model],
                            "dataset": dataset,
                            "condition": condition,
                            "condition_index": condition_index,
                            "example_id": example["example_id"],
                            "sample_ordinal": example["sample_ordinal"],
                            "array_index": array_index,
                            "shard_index": shard_index,
                            "num_shards": resources.shards_per_condition,
                            "production_manifest_path": str(manifest),
                        }
                    )
    _reject_duplicate_run_ids(rows)
    out_path = production_root / "expected_runs" / "v3_expected_runs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    array_tasks = sum(
        len(production_conditions(multihoprag_oracle_allowed=multihoprag_oracle_allowed, dataset=dataset))
        * resources.shards_per_condition
        for dataset in DATASETS
        for resources in MODEL_RESOURCES.values()
    )
    return {"path": out_path, "n": len(rows), "array_tasks": array_tasks}


def render_sbatch(*, production_root: Path, dataset: str, model: str, conditions: tuple[str, ...]) -> str:
    resources = MODEL_RESOURCES[model]
    script_name = SCRIPT_NAMES[(dataset, model)]
    manifest = _manifest_path(production_root, dataset)
    corpus = production_root / "corpus" / f"{MANIFEST_DATASET_NAMES[dataset]}_corpus.jsonl"
    evidence = production_root / "evidence" / f"{MANIFEST_DATASET_NAMES[dataset]}_evidence.jsonl"
    out_dir = production_root / "logs" / model / dataset
    task_count = len(conditions) * resources.shards_per_condition
    conditions_literal = " ".join(shlex.quote(condition) for condition in conditions)
    reasoning_parser_args = ""
    if resources.reasoning_parser:
        reasoning_parser_args = f"  --reasoning-parser {shlex.quote(resources.reasoning_parser)} \\\n"
    tool_parser_plugin_args = ""
    if resources.tool_parser_plugin:
        plugin_path = REPO_ROOT / resources.tool_parser_plugin
        tool_parser_plugin_args = f"  --tool-parser-plugin {shlex.quote(str(plugin_path))} \\\n"
    return f"""#!/bin/bash
# Generated by attack.launch.generate_production_arrays
#SBATCH --job-name=v3_{dataset}_{model}
#SBATCH --partition={resources.partition}
#SBATCH --gres={resources.gres}
#SBATCH --nodes=1
#SBATCH --nodelist={resources.nodelist}
#SBATCH --cpus-per-task={resources.cpus}
#SBATCH --mem={resources.mem}
#SBATCH --time={resources.wall_time}
#SBATCH --array=0-{task_count - 1}%{resources.initial_throttle}
#SBATCH --signal=B:USR1@300
#SBATCH --output={out_dir}/%x_%A_%a.out
#SBATCH --error={out_dir}/%x_%A_%a.err

set -euo pipefail
echo "=== array {script_name} job=${{SLURM_JOB_ID}} task=${{SLURM_ARRAY_TASK_ID}} $(date) ==="

REPO={shlex.quote(str(REPO_ROOT))}
PRODUCTION_ROOT={shlex.quote(str(production_root))}
MODEL={shlex.quote(model)}
MODEL_ID={shlex.quote(MODEL_IDS[model])}
MAX_MODEL_LEN={resources.max_model_len}
MODEL_SEED=20260820
GATE_CONTRACT_VERSION=v3_paired_replay_anchor_20260826
DATASET={shlex.quote(dataset)}
NUM_SHARDS={resources.shards_per_condition}
CONDITIONS=({conditions_literal})
CONDITION_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
CONDITION="${{CONDITIONS[$CONDITION_INDEX]}}"
MANIFEST={shlex.quote(str(manifest))}
CORPUS={shlex.quote(str(corpus))}
EVIDENCE={shlex.quote(str(evidence))}
DEV_CORPUS={shlex.quote(str(DEV_BUNDLE / "dev_corpus.jsonl"))}
DEV_EVIDENCE={shlex.quote(str(DEV_BUNDLE / "dev_evidence.jsonl"))}
DEV_EXAMPLES={shlex.quote(str(DEV_BUNDLE / "smoke_examples.jsonl"))}
GATE_DIR="$PRODUCTION_ROOT/runtime_gates/$MODEL/$DATASET"
LOG_DIR={shlex.quote(str(out_dir))}
mkdir -p "$LOG_DIR" "$GATE_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate {VLLM_CONDA_ENV}
set -u

# Triton/xgrammar JIT compiles CUDA helper modules at vLLM startup and first
# structured-output use. The vllm conda activation exports a broken conda
# cross-compiler, while the loaded Spack GCC can find cc1 but not its runtime
# objects on these nodes. Pin a complete RHEL system GCC toolchain instead.
SYSTEM_GCC=/usr/bin/gcc
SYSTEM_GXX=/usr/bin/g++
SYSTEM_GCC_VERSION=11
SYSTEM_GCC_TARGET=x86_64-redhat-linux
SYSTEM_GCC_LIB=/usr/lib/gcc/$SYSTEM_GCC_TARGET/$SYSTEM_GCC_VERSION
SYSTEM_GCC_LIBEXEC=/usr/libexec/gcc/$SYSTEM_GCC_TARGET/$SYSTEM_GCC_VERSION
if [[ ! -x "$SYSTEM_GCC" || ! -x "$SYSTEM_GXX" || ! -x "$SYSTEM_GCC_LIBEXEC/cc1" || ! -f "$SYSTEM_GCC_LIB/crtbeginS.o" || ! -f "$SYSTEM_GCC_LIB/libgcc.a" ]]; then
  echo "System GCC toolchain is incomplete: CC=$SYSTEM_GCC CXX=$SYSTEM_GXX lib=$SYSTEM_GCC_LIB libexec=$SYSTEM_GCC_LIBEXEC" >&2
  exit 3
fi
export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
unset GCC_EXEC_PREFIX CPATH
export CUDA_HOME=/cm/local/apps/cuda/libs/current
export CC="$SYSTEM_GCC"
export CXX="$SYSTEM_GXX"
export GCC="$SYSTEM_GCC"
export LD=/usr/bin/ld
export AS=/usr/bin/as
export AR=/usr/bin/ar
export COMPILER_PATH="$SYSTEM_GCC_LIBEXEC:$SYSTEM_GCC_LIB"
export LIBRARY_PATH="$SYSTEM_GCC_LIB:/usr/lib64:/usr/lib"
export C_INCLUDE_PATH="$SYSTEM_GCC_LIB/include:/usr/include"
export CPLUS_INCLUDE_PATH="$SYSTEM_GCC_LIB/include:/usr/include/c++/$SYSTEM_GCC_VERSION:/usr/include/c++/$SYSTEM_GCC_VERSION/$SYSTEM_GCC_TARGET:/usr/include"
PREV_LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:/usr/lib64:/usr/lib:$PREV_LD_LIBRARY_PATH"
CC1_PATH="$("$CC" -print-prog-name=cc1 || true)"
CRTBEGIN_PATH="$("$CC" -print-file-name=crtbeginS.o || true)"
LIBGCC_PATH="$("$CC" -print-libgcc-file-name || true)"
if [[ "$CC1_PATH" == "cc1" || ! -x "$CC1_PATH" || "$CRTBEGIN_PATH" == "crtbeginS.o" || ! -f "$CRTBEGIN_PATH" || "$LIBGCC_PATH" == "libgcc.a" || ! -f "$LIBGCC_PATH" ]]; then
  echo "Configured CC cannot find executable cc1: CC=$CC cc1=$CC1_PATH" >&2
  echo "Configured CC cannot find runtime objects: crtbeginS=$CRTBEGIN_PATH libgcc=$LIBGCC_PATH" >&2
  exit 3
fi
export TRITON_CACHE_DIR="$PRODUCTION_ROOT/triton_cache/$MODEL/$DATASET/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$TRITON_CACHE_DIR"
COMPILER_SMOKE_DIR="$(mktemp -d /tmp/v3_compiler_smoke.XXXXXX)"
printf '#include <Python.h>\\nint v3_compiler_smoke(void){{return 0;}}\\n' > "$COMPILER_SMOKE_DIR/smoke.c"
"$CC" "$COMPILER_SMOKE_DIR/smoke.c" -O3 -shared -fPIC -Wno-psabi \\
  -o "$COMPILER_SMOKE_DIR/smoke.so" \\
  -l:libcuda.so.1 \\
  -L/home/sbhengesa/.conda/envs/vllm/lib/python3.11/site-packages/triton/backends/nvidia/lib \\
  -L"$CUDA_HOME/lib64" \\
  -I/home/sbhengesa/.conda/envs/vllm/include/python3.11 \\
  -I/home/sbhengesa/.conda/envs/vllm/lib/python3.11/site-packages/triton/backends/nvidia/include
rm -rf "$COMPILER_SMOKE_DIR"
python - <<'PY'
import importlib.util

missing = [name for name in ("torch", "triton", "vllm", "xgrammar") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing vLLM runtime modules: {{missing}}")
print("vLLM runtime module import smoke passed")
PY
echo "Using compiler for Triton/xgrammar JIT: CC=$CC CXX=$CXX cc1=$CC1_PATH crtbeginS=$CRTBEGIN_PATH libgcc=$LIBGCC_PATH TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

SERVER_PID=""
trap 'if [[ -n "${{SERVER_PID:-}}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
find_slurm_command() {{
  local name="$1"
  local path
  path="$(command -v "$name" 2>/dev/null || true)"
  if [[ -n "$path" ]]; then
    printf '%s\n' "$path"
    return 0
  fi
  for path in /cm/shared/apps/slurm/current/bin/"$name" /cm/local/apps/slurm/current/bin/"$name" /usr/bin/"$name" /bin/"$name"; do
    if [[ -x "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}}
SCANCEL_BIN="$(find_slurm_command scancel || true)"
SCONTROL_BIN="$(find_slurm_command scontrol || true)"
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
VLLM_LOG="$LOG_DIR/vllm_${{MODEL}}_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.log"
python -m vllm.entrypoints.openai.api_server \\
  --model "$MODEL_ID" \\
  --served-model-name "$MODEL_ID" \\
  --host 127.0.0.1 \\
  --port "$PORT" \\
  --dtype bfloat16 \\
  --gpu-memory-utilization {resources.gpu_memory_utilization} \\
  --max-model-len "$MAX_MODEL_LEN" \\
  --tensor-parallel-size {resources.tensor_parallel_size} \\
  --generation-config vllm \\
  --enable-auto-tool-choice \\
{tool_parser_plugin_args}\
  --tool-call-parser {resources.tool_call_parser} \\
{reasoning_parser_args}\
  --enforce-eager \\
  > "$VLLM_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for attempt in $(seq 1 $(( ({resources.vllm_ready_timeout_sec} + 4) / 5 ))); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${{PORT}}/v1/models', timeout=2).read()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 200 "$VLLM_LOG" >&2 || true
    exit 2
  fi
  sleep 5
done
if [[ "$READY" != "1" ]]; then
  tail -n 200 "$VLLM_LOG" >&2 || true
  exit 2
fi

set +u
conda activate {ODR_CONDA_ENV}
set -u
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/open_deep_research/src"
export LANGSMITH_TRACING=false

exec 9>"$GATE_DIR/RUNTIME_GATE.lock"
flock 9
if [[ -f "$GATE_DIR/RUNTIME_GATE_PASSED.json" ]]; then
  python -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); actual=int(data.get("checks", {{}}).get("max_model_len") or -1); expected=int(sys.argv[2]); contract=str(data.get("gate_contract_version") or ""); expected_contract=sys.argv[3]; raise SystemExit(0 if data.get("passed") and actual == expected and contract == expected_contract else 1)' "$GATE_DIR/RUNTIME_GATE_PASSED.json" "$MAX_MODEL_LEN" "$GATE_CONTRACT_VERSION" || rm -f "$GATE_DIR/RUNTIME_GATE_PASSED.json" "$GATE_DIR/RUNTIME_GATE_FAILED.json"
fi
if [[ ! -f "$GATE_DIR/RUNTIME_GATE_PASSED.json" ]]; then
  rm -f "$GATE_DIR/RUNTIME_GATE_FAILED.json"
  set +e
  python -B -m attack.runtime.runtime_gate \\
    --model "$MODEL" \\
    --model-id "$MODEL_ID" \\
    --model-base-url "http://127.0.0.1:${{PORT}}/v1" \\
    --dataset "$DATASET" \\
    --corpus "$DEV_CORPUS" \\
    --evidence "$DEV_EVIDENCE" \\
    --examples "$DEV_EXAMPLES" \\
    --gate-dir "$GATE_DIR" \\
    --tokenizer-python {shlex.quote(str(WHITEBOX_PYTHON))} \\
    --tool-call-parser {resources.tool_call_parser} \\
    --max-model-len "$MAX_MODEL_LEN" \\
    --model-seed "$MODEL_SEED" \\
    --top-k 3 \\
    --research-model-max-tokens {resources.research_model_max_tokens} \\
    --compression-model-max-tokens {resources.compression_model_max_tokens} \\
    --final-report-model-max-tokens {resources.final_report_model_max_tokens}
  GATE_RC=$?
  set -e
  if [[ "$GATE_RC" -ne 0 ]]; then
    echo "runtime gate command exited $GATE_RC; checking gate report files" >&2
  fi
fi
if [[ ! -f "$GATE_DIR/RUNTIME_GATE_PASSED.json" ]]; then
  echo "runtime gate failed; preserving logs in $GATE_DIR and $LOG_DIR" >&2
  if [[ -n "${{SLURM_ARRAY_JOB_ID:-}}" && -n "${{SLURM_ARRAY_TASK_ID:-}}" && -n "${{SLURM_ARRAY_TASK_MAX:-}}" ]]; then
    NEXT_TASK=$((SLURM_ARRAY_TASK_ID + 1))
    if (( NEXT_TASK <= SLURM_ARRAY_TASK_MAX )); then
      if [[ -n "$SCANCEL_BIN" ]]; then
        "$SCANCEL_BIN" "${{SLURM_ARRAY_JOB_ID}}_[${{NEXT_TASK}}-${{SLURM_ARRAY_TASK_MAX}}]" || true
      else
        echo "warning: scancel not found; unable to cancel remaining array tasks" >&2
      fi
    fi
    if [[ -n "$SCONTROL_BIN" ]]; then
      "$SCONTROL_BIN" hold "${{SLURM_ARRAY_JOB_ID}}" || true
    else
      echo "warning: scontrol not found; unable to hold failed array" >&2
    fi
  fi
  exit 20
fi
flock -u 9

python -B -m attack.runtime.production_runner \\
  --model "$MODEL" \\
  --model-id "$MODEL_ID" \\
  --dataset "$DATASET" \\
  --condition "$CONDITION" \\
  --manifest "$MANIFEST" \\
  --corpus "$CORPUS" \\
  --evidence "$EVIDENCE" \\
  --out-root "$PRODUCTION_ROOT" \\
  --array-index "$SLURM_ARRAY_TASK_ID" \\
  --shard-index "$SHARD_INDEX" \\
  --num-shards "$NUM_SHARDS" \\
  --model-base-url "http://127.0.0.1:${{PORT}}/v1" \\
  --model-seed "$MODEL_SEED" \\
  --research-model-max-tokens {resources.research_model_max_tokens} \\
  --compression-model-max-tokens {resources.compression_model_max_tokens} \\
  --final-report-model-max-tokens {resources.final_report_model_max_tokens} \\
  --top-k 3 \\
  --abort-after-consecutive-errors 3
"""


def render_submit_wrapper(*, production_root: Path, scripts: list[str]) -> str:
    scripts_literal = "\n".join(f"  {shlex.quote(script)}" for script in scripts)
    return f"""#!/bin/bash
set -euo pipefail
REPO={shlex.quote(str(REPO_ROOT))}
PRODUCTION_ROOT={shlex.quote(str(production_root))}
cd "$REPO"

{shlex.quote(str(WHITEBOX_PYTHON))} -B -m attack.launch.generate_production_arrays --mode validate --production-root "$PRODUCTION_ROOT" --local-files-only

SCRIPTS=(
{scripts_literal}
)
SUBMISSION_DIR="$PRODUCTION_ROOT/submissions"
mkdir -p "$SUBMISSION_DIR"
SUBMISSION_JSON="$SUBMISSION_DIR/submission_$(date +%Y%m%d_%H%M%S).jsonl"
for script in "${{SCRIPTS[@]}}"; do
  job_name=$(awk -F= '/^#SBATCH --job-name=/ {{print $2; exit}}' "$script")
  if JOB_NAME="$job_name" {shlex.quote(str(WHITEBOX_PYTHON))} -B -c 'import os, subprocess; user=os.environ.get("USER",""); cmd=["squeue","-h","-o","%j"]; cmd[1:1]=["-u", user] if user else []; out=subprocess.run(cmd, capture_output=True, text=True); names={{line.strip() for line in out.stdout.splitlines() if line.strip()}}; raise SystemExit(0 if os.environ["JOB_NAME"] in names else 1)'; then
    JOB_NAME="$job_name" SCRIPT="$script" {shlex.quote(str(WHITEBOX_PYTHON))} -B -c 'import json, os, time; print(json.dumps({{"script": os.environ["SCRIPT"], "job_name": os.environ["JOB_NAME"], "skipped": True, "reason": "already_active", "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}, sort_keys=True))' | tee -a "$SUBMISSION_JSON"
    continue
  fi
  job_id=$(sbatch --parsable "$script")
  JOB_ID="$job_id" JOB_NAME="$job_name" SCRIPT="$script" {shlex.quote(str(WHITEBOX_PYTHON))} -B -c 'import json, os, time; print(json.dumps({{"script": os.environ["SCRIPT"], "job_name": os.environ["JOB_NAME"], "job_id": os.environ["JOB_ID"], "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}, sort_keys=True))' | tee -a "$SUBMISSION_JSON"
done
job_ids=$({shlex.quote(str(WHITEBOX_PYTHON))} -B -c 'import json, sys; print(",".join(json.loads(line).get("job_id", "") for line in open(sys.argv[1], encoding="utf-8") if line.strip() and json.loads(line).get("job_id")))' "$SUBMISSION_JSON")
if [[ -n "$job_ids" ]]; then
  squeue -j "$job_ids" > "$SUBMISSION_DIR/initial_squeue.txt" || true
fi
echo "$SUBMISSION_JSON"
"""


def validate_artifacts(*, production_root: Path, local_files_only: bool, run_tests: bool = True) -> dict:
    checks: dict[str, object] = {}
    failures = []
    if "v1" in str(production_root) or "v2" in str(production_root):
        failures.append(f"production root points to v1/v2 path: {production_root}")

    manifest_reports = []
    for dataset in DATASETS:
        manifest = _manifest_path(production_root, dataset)
        if not manifest.exists():
            failures.append(f"missing manifest: {manifest}")
            continue
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [row["example_id"] for row in rows]
        if len(rows) != N_SAMPLE or len(set(ids)) != N_SAMPLE:
            failures.append(f"{dataset} manifest count/uniqueness failed: {len(rows)} rows, {len(set(ids))} unique")
        if any(str(example_id).startswith("dev_") for example_id in ids):
            failures.append(f"{dataset} manifest contains development IDs")
        manifest_reports.append({"dataset": dataset, "rows": len(rows), "sha256": _file_sha256(manifest)})
    checks["manifests"] = manifest_reports

    cue_rows = cue_registry_report()
    bad_cues = [row for row in cue_rows if not row["production_valid"]]
    if bad_cues:
        failures.append(f"invalid production cues: {bad_cues}")
    checks["cue_registry"] = cue_rows

    tokenizer_report = tokenizer_counts_report()
    checks["tokenizer_counts"] = tokenizer_report
    if not tokenizer_report["valid"]:
        failures.append(f"tokenizer cue length validation failed: {tokenizer_report}")
    checks["frozen_pipeline_fingerprint"] = frozen_pipeline_fingerprint(
        production_root=production_root,
        manifest_reports=manifest_reports,
        tokenizer_report=tokenizer_report,
        cue_rows=cue_rows,
    )

    oracle_audit = multihoprag_oracle_audit(production_root)
    checks["multihoprag_oracle_audit"] = oracle_audit

    matrix = build_expected_run_manifest(production_root=production_root, multihoprag_oracle_allowed=oracle_audit["allowed"])
    checks["expected_runs"] = {"path": str(matrix["path"]), "n": matrix["n"], "array_tasks": matrix["array_tasks"]}
    expected_runs = len(FINAL_PRODUCTION_MODELS) * sum(
        N_SAMPLE * len(production_conditions(multihoprag_oracle_allowed=oracle_audit["allowed"], dataset=dataset))
        for dataset in DATASETS
    )
    expected_tasks = sum(
        len(production_conditions(multihoprag_oracle_allowed=oracle_audit["allowed"], dataset=dataset))
        * MODEL_RESOURCES[model].shards_per_condition
        for dataset in DATASETS
        for model in FINAL_PRODUCTION_MODELS
    )
    if matrix["n"] != expected_runs or matrix["array_tasks"] != expected_tasks:
        failures.append(f"run expansion mismatch: {matrix}; expected {expected_runs}/{expected_tasks}")
    expected_run_models = _expected_run_models(Path(matrix["path"]))
    checks["expected_run_models"] = expected_run_models
    if expected_run_models != list(FINAL_PRODUCTION_MODELS):
        failures.append(f"final expected-run manifest has wrong models: {expected_run_models}")
    if any(model in PIPELINE_VALIDATION_ONLY_MODEL_IDS for model in expected_run_models):
        failures.append(f"pipeline-validation-only model leaked into final expected runs: {expected_run_models}")

    scripts = [production_root / "sbatch" / SCRIPT_NAMES[(dataset, model)] for dataset in DATASETS for model in FINAL_PRODUCTION_MODELS]
    checks["scripts"] = []
    for script in scripts:
        if not script.exists():
            failures.append(f"missing script: {script}")
            continue
        text = script.read_text(encoding="utf-8")
        if "qwen2.5-7b" in text.lower() or "qwen7b" in text.lower():
            failures.append(f"Qwen-7B appears in final-production script {script}")
        if "tavily" in text.lower() or "duckduckgo" in text.lower() or "ddg" in text.lower():
            failures.append(f"external retrieval marker appears in {script}")
        required_vllm_compiler_markers = [
            "SYSTEM_GCC=/usr/bin/gcc",
            'export CC="$SYSTEM_GCC"',
            'export COMPILER_PATH="$SYSTEM_GCC_LIBEXEC:$SYSTEM_GCC_LIB"',
            'export LIBRARY_PATH="$SYSTEM_GCC_LIB:/usr/lib64:/usr/lib"',
            'export C_INCLUDE_PATH="$SYSTEM_GCC_LIB/include:/usr/include"',
            'CC1_PATH="$("$CC" -print-prog-name=cc1 || true)"',
            'CRTBEGIN_PATH="$("$CC" -print-file-name=crtbeginS.o || true)"',
            "Configured CC cannot find executable cc1",
            "Configured CC cannot find runtime objects",
            "v3_compiler_smoke",
            "-l:libcuda.so.1",
            "vLLM runtime module import smoke passed",
            'export TRITON_CACHE_DIR="$PRODUCTION_ROOT/triton_cache/$MODEL/$DATASET/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"',
        ]
        missing_compiler_markers = [marker for marker in required_vllm_compiler_markers if marker not in text]
        if missing_compiler_markers:
            failures.append(f"missing vLLM compiler guard markers in {script}: {missing_compiler_markers}")
        syntax = _run(["bash", "-n", str(script)])
        if syntax["returncode"] != 0:
            failures.append(f"shell syntax failed for {script}: {syntax['stderr']}")
        test_only = _run(["sbatch", "--test-only", str(script)])
        checks["scripts"].append({"path": str(script), "bash_n": syntax, "sbatch_test_only": test_only})
        if test_only["returncode"] != 0 and "unrecognized option" not in (test_only["stderr"] + test_only["stdout"]).lower():
            failures.append(f"sbatch --test-only failed for {script}: {test_only['stderr'] or test_only['stdout']}")

    if run_tests:
        checks["unit_and_real_schema_tests"] = _run([str(WHITEBOX_PYTHON), "-B", "-m", "unittest", "discover", "-s", "attack/tests"])
        if checks["unit_and_real_schema_tests"]["returncode"] != 0:
            failures.append("unit/real-schema tests failed")
        checks["odr_graph_compile"] = _run([str(ODR_PYTHON), "-B", "-c", "import sys; sys.path.insert(0, 'open_deep_research/src'); from open_deep_research.deep_researcher import deep_researcher_builder; deep_researcher_builder.compile(); print('ok')"])
        if checks["odr_graph_compile"]["returncode"] != 0:
            failures.append("ODR graph compilation failed")

    checks["retriever_calibration"] = _calibration_summary()
    checks["bm25_determinism"] = {"covered_by_test": "test_bm25_is_deterministic_with_doc_id_tiebreak"}
    checks["shared_prefix_hashing"] = {
        "covered_by_test": "test_paired_trace_validation_accepts_shared_base; "
        "test_shared_prefix_hash_ignores_runtime_tool_ids"
    }
    checks["trace_schema"] = {"covered_by_test": "validate_branch_trace and validate_paired_traces tests"}
    checks["resume_idempotency"] = {"stable_run_ids": True, "result_rows_skip_existing_ok_runs": True}
    checks["timeout_censoring"] = {
        "slurm_signal": "B:USR1@300",
        "runner_marks_non_ok_rows_censored_or_timeout": True,
    }

    report = {
        "schema_version": "v3_prelaunch_report_v1",
        "production_root": str(production_root),
        "local_files_only": local_files_only,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "report_created_at": int(time.time()),
    }
    report["report_sha256"] = stable_hash({k: v for k, v in report.items() if k != "report_sha256"}, length=None)
    report_path = production_root / "prelaunch" / "PRELAUNCH_REPORT.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    return report


def cue_registry_report() -> list[dict]:
    rows = []
    for condition in V3_CONDITIONS:
        cue = suffix_for_condition(condition)
        rows.append(
            {
                "condition": condition,
                "suffix": cue.suffix,
                "suffix_sha256": cue.suffix_hash,
                "whitespace_word_count": len(cue.suffix.split()) if cue.suffix else 0,
                "requires_injection": cue.requires_injection,
                "oracle_complete": cue.oracle_complete,
                "production_valid": condition not in DEPRECATED_CONDITION_ALIASES and "Best" not in condition,
            }
        )
    return rows


def tokenizer_counts_report() -> dict:
    from transformers import AutoTokenizer

    strings = {
        "NaturalNeutral": NEUTRAL_SUFFIX,
        "NaturalContinueSearch": CONTINUE_SEARCH_SUFFIX,
        "Natural_PreRegisteredSufficiency": PREREGISTERED_SUFFICIENCY_SUFFIX,
        "Natural_PreRegisteredSufficiencySaturation": PREREGISTERED_SUFFICIENCY_SATURATION_SUFFIX,
    }
    by_model = {}
    tokenizer_hashes = {}
    for model, model_id in MODEL_IDS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        tokenizer_hash = stable_hash(
            {
                "model_id": model_id,
                "vocab_size": len(tokenizer),
                "chat_template": chat_template,
            },
            length=None,
        )
        tokenizer_hashes[model] = tokenizer_hash
        by_model[model] = {
            "model_id": model_id,
            "vocab_size": len(tokenizer),
            "chat_template_hash": hash_text(chat_template, length=None) if chat_template else "",
            "has_chat_template": bool(chat_template),
            "suffix_token_counts": {
            name: {
                "whitespace_words": len(text.split()),
                "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                "sha256": hash_text(text, length=None),
            }
            for name, text in strings.items()
            },
        }
    max_diff = 0
    for name in strings:
        counts = [by_model[model]["suffix_token_counts"][name]["token_count"] for model in MODEL_IDS]
        max_diff = max(max_diff, max(counts) - min(counts))
    return {
        "models": by_model,
        "tokenizer_hashes": tokenizer_hashes,
        "shared_tokenizer_hash": len(set(tokenizer_hashes.values())) == 1,
        "max_between_model_token_count_difference": max_diff,
        "valid": set(by_model) == set(FINAL_PRODUCTION_MODELS) and all(row["has_chat_template"] for row in by_model.values()),
    }


def frozen_pipeline_fingerprint(
    *,
    production_root: Path,
    manifest_reports: Iterable[Mapping[str, object]] | None = None,
    tokenizer_report: Mapping[str, object] | None = None,
    cue_rows: Iterable[Mapping[str, object]] | None = None,
) -> dict:
    manifest_reports = list(manifest_reports or _manifest_file_reports(production_root))
    cue_rows = list(cue_rows or cue_registry_report())
    tokenizer_report = dict(tokenizer_report or tokenizer_counts_report())
    corpus_reports = [
        _path_report(production_root / "corpus" / f"{MANIFEST_DATASET_NAMES[dataset]}_corpus.jsonl")
        for dataset in DATASETS
    ]
    evidence_reports = [
        _path_report(production_root / "evidence" / f"{MANIFEST_DATASET_NAMES[dataset]}_evidence.jsonl")
        for dataset in DATASETS
    ]
    source_groups = {
        "odr_graph_code": [
            "open_deep_research/src/open_deep_research/deep_researcher.py",
            "open_deep_research/src/open_deep_research/state.py",
            "open_deep_research/src/open_deep_research/configuration.py",
        ],
        "system_and_agent_prompts": [
            "open_deep_research/src/open_deep_research/prompts.py",
        ],
        "tool_schemas": [
            "open_deep_research/src/open_deep_research/state.py",
            "open_deep_research/src/open_deep_research/utils.py",
            "attack/runtime/benchmark_search.py",
        ],
        "local_retriever_code": [
            "attack/runtime/benchmark_search.py",
            "attack/runtime/corpus.py",
            "attack/runtime/evidence.py",
        ],
        "attack_suffixes_and_conditions": [
            "attack/runtime/cues.py",
            "attack/cue_definitions.py",
        ],
        "runner_trace_and_final_answer_writer": [
            "attack/runtime/runner.py",
            "attack/runtime/production_runner.py",
            "attack/runtime/trace_schema.py",
        ],
        "production_manifest_and_model_selection": [
            "attack/runtime/production_manifest.py",
            "attack/runtime/production_models.py",
            "attack/launch/generate_production_arrays.py",
        ],
    }
    source_hashes = {
        name: [_path_report(REPO_ROOT / relative) for relative in relatives]
        for name, relatives in source_groups.items()
    }
    model_configs = {
        model: {
            "model_id": MODEL_IDS[model],
            "resources": asdict(MODEL_RESOURCES[model]),
            "tool_call_parser": MODEL_TOOL_CALL_PARSERS[model],
            "reasoning_parser": MODEL_REASONING_PARSERS.get(model, ""),
            "temperature": 0.0,
            "seed": 20260820,
            "top_k": 3,
        }
        for model in FINAL_PRODUCTION_MODELS
    }
    payload = {
        "schema_version": "v3_frozen_pipeline_fingerprint_v1",
        "victim_system_components_intentionally_unmodified": True,
        "source_hashes": source_hashes,
        "tool_call_parsers": dict(MODEL_TOOL_CALL_PARSERS),
        "reasoning_parsers": dict(MODEL_REASONING_PARSERS),
        "model_configs": model_configs,
        "tokenizers": tokenizer_report,
        "condition_manifest": {
            "conditions": list(V3_CONDITIONS),
            "cue_rows": cue_rows,
            "sha256": stable_hash({"conditions": list(V3_CONDITIONS), "cue_rows": cue_rows}, length=None),
        },
        "dataset_manifests": list(manifest_reports),
        "corpus": corpus_reports,
        "evidence": evidence_reports,
    }
    payload["combined_sha256"] = stable_hash(payload, length=None)
    return payload


def _manifest_file_reports(production_root: Path) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "path": str(_manifest_path(production_root, dataset)),
            "sha256": _file_sha256(_manifest_path(production_root, dataset)) if _manifest_path(production_root, dataset).exists() else "",
        }
        for dataset in DATASETS
    ]


def _path_report(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path) if path.exists() else "",
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _expected_run_models(path: Path) -> list[str]:
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(str(json.loads(line).get("model") or ""))
    ordered = [model for model in FINAL_PRODUCTION_MODELS if model in seen]
    ordered.extend(sorted(model for model in seen if model and model not in FINAL_PRODUCTION_MODELS))
    return ordered


def submit_arrays(*, production_root: Path) -> dict:
    validate_artifacts(production_root=production_root, local_files_only=True, run_tests=True)
    scripts = [production_root / "sbatch" / SCRIPT_NAMES[(dataset, model)] for dataset in DATASETS for model in FINAL_PRODUCTION_MODELS]
    active_names = _active_job_names()
    rows = []
    skipped = []
    for script in scripts:
        job_name = _script_job_name(script)
        if job_name in active_names:
            skipped.append({"script": str(script), "job_name": job_name, "reason": "already_active"})
            continue
        result = _run(["sbatch", "--parsable", str(script)])
        if result["returncode"] != 0:
            raise RuntimeError(f"sbatch failed for {script}: {result}")
        job_id = result["stdout"].strip().splitlines()[-1]
        rows.append({"script": str(script), "job_name": job_name, "job_id": job_id})
    job_ids = ",".join(row["job_id"] for row in rows)
    state = _run(["squeue", "-j", job_ids]) if job_ids else {"cmd": ["squeue", "-j", ""], "returncode": 0, "stdout": "", "stderr": ""}
    report = {
        "schema_version": "v3_submission_v1",
        "jobs": rows,
        "skipped_active_jobs": skipped,
        "initial_squeue": state,
    }
    out = production_root / "submissions" / f"submission_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _script_job_name(script: Path) -> str:
    for line in script.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH --job-name="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"missing #SBATCH --job-name in {script}")


def _active_job_names() -> set[str]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    cmd = ["squeue", "-h", "-o", "%j"]
    if user:
        cmd[1:1] = ["-u", user]
    state = _run(cmd)
    if state["returncode"] != 0:
        return set()
    return {line.strip() for line in str(state["stdout"]).splitlines() if line.strip()}


def _manifest_path(production_root: Path, dataset: str) -> Path:
    return production_root / "manifests" / f"{MANIFEST_DATASET_NAMES[dataset]}_n450.jsonl"


def _reject_duplicate_run_ids(rows: Iterable[Mapping[str, object]]) -> None:
    seen = set()
    duplicates = []
    for row in rows:
        run_id = str(row["run_id"])
        if run_id in seen:
            duplicates.append(run_id)
        seen.add(run_id)
    if duplicates:
        raise ValueError(f"duplicate run ids: {duplicates[:10]}")


def _run(cmd: list[str]) -> dict[str, object]:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    return {"cmd": cmd, "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


def _calibration_summary() -> dict:
    path = REPO_ROOT / "results" / "nonproduction_validation" / "retriever_calibration.json"
    if not path.exists():
        return {"available": False}
    rows = json.loads(path.read_text(encoding="utf-8"))
    top3 = next((row for row in rows if row.get("top_k") == 3), None)
    return {"available": True, "recommended_top_k": 3, "top_k_3": top3}


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "generate", "validate", "submit", "all"), default="all")
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "freeze":
        print(json.dumps(freeze_all_manifests(production_root=args.production_root, local_files_only=args.local_files_only), indent=2, sort_keys=True))
    elif args.mode == "generate":
        print(json.dumps(generate_artifacts(production_root=args.production_root, local_files_only=args.local_files_only), indent=2, sort_keys=True))
    elif args.mode == "validate":
        print(json.dumps(validate_artifacts(production_root=args.production_root, local_files_only=args.local_files_only, run_tests=not args.skip_tests), indent=2, sort_keys=True))
    elif args.mode == "submit":
        print(json.dumps(submit_arrays(production_root=args.production_root), indent=2, sort_keys=True))
    elif args.mode == "all":
        generate_artifacts(production_root=args.production_root, local_files_only=args.local_files_only)
        validate_artifacts(production_root=args.production_root, local_files_only=args.local_files_only, run_tests=not args.skip_tests)
        print(json.dumps(submit_arrays(production_root=args.production_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Generate SLURM scripts for the paper PCR rerun.

Pipeline:
  1. Freeze canonical samples per dataset.
  2. Run clean baselines for every model/dataset.
  3. Build global eligible manifests by intersecting clean-eligible IDs across
     all configured models.
  4. Run WGA, FactOnly, and CueOnly on those global eligible manifests.

All generated outputs live under whitebox_paper_results/pcr_full_completion_v1
unless --results-root is supplied.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACK_DIR = REPO_ROOT / "whitebox_attack"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "whitebox_paper_results" / "pcr_full_completion_v1"

DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")

MODELS = {
    "qwen7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "mem": "80G",
        "time": "40:00:00",
    },
    "qwen14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "mem": "80G",
        "time": "48:00:00",
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "mem": "80G",
        "time": "40:00:00",
    },
    "gemma27b": {
        "model_id": "google/gemma-3-27b-it",
        "mem": "80G",
        "time": "48:00:00",
    },
    "gptoss20b": {
        "model_id": "openai/gpt-oss-20b",
        "mem": "96G",
        "time": "48:00:00",
        "device_map": "auto",
    },
}

ATTACKS = {
    "wga": {
        "script": "wrong_gap_assessment.py",
        "prefix": "final_wga",
        "time": None,
        "requires_wga": False,
        "requires_baseline": True,
    },
    "factonly": {
        "script": "wrong_gap_assessment_factonly.py",
        "prefix": "final_wga_factonly",
        "time": "24:00:00",
        "requires_wga": True,
        "requires_baseline": False,
    },
    "cueonly": {
        "script": "wrong_gap_assessment_cue_only.py",
        "prefix": "final_wga_cueonly",
        "time": None,
        "requires_wga": True,
        "requires_baseline": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper rerun sbatch files for white-box PCR experiments."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--attacks", nargs="*", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--n-sample", type=int, default=450)
    parser.add_argument("--stage", default="PAPER_PCR_FULL_COMPLETION_V1")
    parser.add_argument("--partition", default="machinelearning")
    parser.add_argument("--gpu", default="nvidia_h100_nvl:1")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--ddg-sleep", default="1.5")
    return parser.parse_args()


def sample_path(results_root: Path, dataset: str, n_sample: int) -> Path:
    return results_root / "samples" / f"{dataset}_n{n_sample}.jsonl"


def eligible_path(results_root: Path, dataset: str) -> Path:
    return results_root / "manifests" / f"global_eligible_{dataset}.jsonl"


def baseline_path(results_root: Path, model_alias: str, dataset: str) -> Path:
    return (
        results_root
        / "baselines"
        / model_alias
        / dataset
        / f"clean_baseline_{model_alias}_{dataset}.jsonl"
    )


def jsonl_path(results_root: Path, model_alias: str, dataset: str, attack: str) -> Path:
    prefix = ATTACKS[attack]["prefix"]
    return (
        results_root
        / "jsonl"
        / model_alias
        / dataset
        / f"{prefix}_{model_alias}_{dataset}.jsonl"
    )


def sbatch_header(
    *,
    job_name: str,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    log_dir: Path,
    gpu: str | None = None,
) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]
    if gpu:
        lines.append(f"#SBATCH --gres=gpu:{gpu}")
    lines.extend(
        [
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={log_dir}/{job_name}_%j.out",
            f"#SBATCH --error={log_dir}/{job_name}_%j.err",
            "",
            "set -uo pipefail",
            'echo "=== NODE: $(hostname) JOB: $SLURM_JOB_ID $(date) ==="',
        ]
    )
    if gpu:
        lines.append("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tail -1")
    lines.extend(
        [
            "source $(conda info --base)/etc/profile.d/conda.sh",
            "conda activate whitebox",
            "",
        ]
    )
    return lines


def render_prepare_samples(args: argparse.Namespace, results_root: Path) -> str:
    job_name = "pcr_prepare_samples"
    log_dir = results_root / "logs" / "_pipeline"
    lines = sbatch_header(
        job_name=job_name,
        partition=args.partition,
        cpus=2,
        mem="16G",
        time_limit="04:00:00",
        log_dir=log_dir,
    )
    lines.extend(
        [
            f'mkdir -p "{log_dir}" "{results_root / "samples"}"',
            f'cd "{ATTACK_DIR}"',
            f'python "generate_sample_manifest.py" --out-dir "{results_root / "samples"}" '
            f"--n-sample {args.n_sample}",
            'echo "=== DONE rc=$? $(date) ==="',
            "",
        ]
    )
    return "\n".join(lines)


def render_build_manifest(args: argparse.Namespace, results_root: Path) -> str:
    job_name = "pcr_build_global_eligible"
    log_dir = results_root / "logs" / "_pipeline"
    lines = sbatch_header(
        job_name=job_name,
        partition=args.partition,
        cpus=2,
        mem="16G",
        time_limit="02:00:00",
        log_dir=log_dir,
    )
    lines.extend(
        [
            f'mkdir -p "{log_dir}" "{results_root / "manifests"}"',
            f'cd "{ATTACK_DIR}"',
            f'python "build_global_eligible_manifest.py" --results-root "{results_root}" '
            f"--n-sample {args.n_sample}",
            'echo "=== DONE rc=$? $(date) ==="',
            "",
        ]
    )
    return "\n".join(lines)


def render_baseline(
    *,
    args: argparse.Namespace,
    results_root: Path,
    model_alias: str,
    dataset: str,
) -> str:
    model_cfg = MODELS[model_alias]
    job_name = f"pcr_base_{model_alias}_{dataset}"
    log_dir = results_root / "logs" / model_alias / dataset
    out_path = baseline_path(results_root, model_alias, dataset)
    s_path = sample_path(results_root, dataset, args.n_sample)
    ddg_cache_dir = results_root / "ddg_cache"

    lines = sbatch_header(
        job_name=job_name,
        partition=args.partition,
        cpus=args.cpus,
        mem=model_cfg["mem"],
        time_limit=model_cfg["time"],
        log_dir=log_dir,
        gpu=args.gpu,
    )
    lines.extend(
        [
            "pip install -q duckduckgo-search",
            f'mkdir -p "{out_path.parent}" "{log_dir}" "{ddg_cache_dir}"',
            f'if [[ ! -s "{s_path}" ]]; then',
            f'  echo "Missing sample manifest: {s_path}" >&2',
            "  exit 2",
            "fi",
            "",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f'export WB_STAGE="{args.stage}"',
            f'export WB_RUN_ID="${{SLURM_JOB_ID}}_baseline_{model_alias}_{dataset}"',
            f'export WB_MODEL="{model_cfg["model_id"]}"',
            f"export WB_MAX_STEPS={args.max_steps}",
            f"export N_SAMPLE={args.n_sample}",
            f'export DATASET_SRC="{dataset}"',
            f'export WB_SAMPLE_MANIFEST="{s_path}"',
            f'export DDG_CACHE_DIR="{ddg_cache_dir}"',
            f"export DDG_SLEEP={args.ddg_sleep}",
            f'export WB_LOG_PATH="{out_path}"',
        ]
    )
    if "device_map" in model_cfg:
        lines.append(f'export WB_DEVICE_MAP="{model_cfg["device_map"]}"')
    lines.extend(
        [
            "",
            f'cd "{ATTACK_DIR}"',
            'python "run_clean_baseline.py"',
            'echo "=== DONE rc=$? $(date) ==="',
            "",
        ]
    )
    return "\n".join(lines)


def render_attack(
    *,
    args: argparse.Namespace,
    results_root: Path,
    model_alias: str,
    dataset: str,
    attack: str,
) -> str:
    model_cfg = MODELS[model_alias]
    attack_cfg = ATTACKS[attack]

    job_name = f"pcr_{attack}_{model_alias}_{dataset}"
    log_dir = results_root / "logs" / model_alias / dataset
    ddg_cache_dir = results_root / "ddg_cache"
    out_path = jsonl_path(results_root, model_alias, dataset, attack)
    wga_path = jsonl_path(results_root, model_alias, dataset, "wga")
    b_path = baseline_path(results_root, model_alias, dataset)
    s_path = eligible_path(results_root, dataset)
    time_limit = attack_cfg["time"] or model_cfg["time"]

    lines = sbatch_header(
        job_name=job_name,
        partition=args.partition,
        cpus=args.cpus,
        mem=model_cfg["mem"],
        time_limit=time_limit,
        log_dir=log_dir,
        gpu=args.gpu,
    )
    lines.extend(
        [
            "pip install -q duckduckgo-search",
            f'mkdir -p "{out_path.parent}" "{log_dir}" "{ddg_cache_dir}"',
            f'if [[ ! -s "{s_path}" ]]; then',
            f'  echo "Missing global eligible manifest: {s_path}" >&2',
            "  exit 2",
            "fi",
            "",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f'export WB_STAGE="{args.stage}"',
            f'export WB_RUN_ID="${{SLURM_JOB_ID}}_{attack}_{model_alias}_{dataset}"',
            f'export WB_MODEL="{model_cfg["model_id"]}"',
            f"export WB_MAX_STEPS={args.max_steps}",
            f"export N_SAMPLE={args.n_sample}",
            f'export DATASET_SRC="{dataset}"',
            f'export WB_SAMPLE_MANIFEST="{s_path}"',
            f'export DDG_CACHE_DIR="{ddg_cache_dir}"',
            f"export DDG_SLEEP={args.ddg_sleep}",
            f'export WB_LOG_PATH="{out_path}"',
        ]
    )

    if "device_map" in model_cfg:
        lines.append(f'export WB_DEVICE_MAP="{model_cfg["device_map"]}"')

    if attack_cfg["requires_baseline"]:
        lines.extend(
            [
                f'export WB_BASELINE_PATH="{b_path}"',
                'if [[ ! -s "$WB_BASELINE_PATH" ]]; then',
                '  echo "Missing clean baseline: $WB_BASELINE_PATH" >&2',
                "  exit 2",
                "fi",
            ]
        )

    if attack_cfg["requires_wga"]:
        lines.extend(
            [
                f'export WB_WGA_PATH="{wga_path}"',
                'if [[ ! -s "$WB_WGA_PATH" ]]; then',
                '  echo "Missing WGA manifest: $WB_WGA_PATH" >&2',
                "  echo \"Submit and complete the matching pcr_wga job first.\" >&2",
                "  exit 2",
                "fi",
            ]
        )

    lines.extend(
        [
            "",
            f'cd "{ATTACK_DIR}"',
            f'python "{attack_cfg["script"]}"',
            'echo "=== DONE rc=$? $(date) ==="',
            "",
        ]
    )
    return "\n".join(lines)


def write_submit_order(
    results_root: Path,
    *,
    prepare_path: Path,
    build_path: Path,
    baseline_paths: list[Path],
    attack_paths: dict[tuple[str, str, str], Path],
) -> None:
    lines = [
        "# Paper PCR Rerun Submit Order",
        "",
        "These jobs write only under this results tree.",
        "",
        "1. Freeze canonical raw samples.",
        "",
        f"   sbatch {prepare_path}",
        "",
        "2. Run clean baselines for every model/dataset.",
        "",
    ]
    lines.extend(f"   sbatch {path}" for path in baseline_paths)
    lines.extend(
        [
            "",
            "3. Build global eligible manifests from the baseline intersection.",
            "",
            f"   sbatch {build_path}",
            "",
            "4. Run WGA attacks using the global eligible manifests.",
            "",
        ]
    )
    lines.extend(
        f"   sbatch {path}"
        for (attack, _model, _dataset), path in attack_paths.items()
        if attack == "wga"
    )
    lines.extend(
        [
            "",
            "5. Run FactOnly and CueOnly after each matching WGA job.",
            "",
        ]
    )
    lines.extend(
        f"   sbatch {path}"
        for (attack, _model, _dataset), path in attack_paths.items()
        if attack in {"factonly", "cueonly"}
    )
    lines.extend(
        [
            "",
            "The generated submit helper wires these phases with afterok dependencies:",
            "",
            f"   python {results_root / 'submit_pipeline.py'}",
            "",
        ]
    )
    (results_root / "submit_order.md").write_text("\n".join(lines), encoding="utf-8")


def write_submit_helper(
    results_root: Path,
    *,
    prepare_path: Path,
    build_path: Path,
    baseline_paths: list[Path],
    attack_paths: dict[tuple[str, str, str], Path],
) -> None:
    baseline_literals = [os.fspath(path) for path in baseline_paths]
    wga_literals = {
        f"{model}:{dataset}": os.fspath(path)
        for (attack, model, dataset), path in attack_paths.items()
        if attack == "wga"
    }
    factonly_literals = {
        f"{model}:{dataset}": os.fspath(path)
        for (attack, model, dataset), path in attack_paths.items()
        if attack == "factonly"
    }
    cueonly_literals = {
        f"{model}:{dataset}": os.fspath(path)
        for (attack, model, dataset), path in attack_paths.items()
        if attack == "cueonly"
    }

    helper = f'''#!/usr/bin/env python3
import argparse
import re
import subprocess

PREPARE = {os.fspath(prepare_path)!r}
BUILD = {os.fspath(build_path)!r}
BASELINES = {baseline_literals!r}
WGA = {wga_literals!r}
FACTONLY = {factonly_literals!r}
CUEONLY = {cueonly_literals!r}


def submit(path, dependency=None, dry_run=False):
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{{dependency}}")
    cmd.append(path)
    print(" ".join(cmd), flush=True)
    if dry_run:
        submit.counter += 1
        return f"DRY{{submit.counter}}"
    out = subprocess.check_output(cmd, text=True)
    print(out.strip(), flush=True)
    match = re.search(r"Submitted batch job (\\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {{out!r}}")
    return match.group(1)


submit.counter = 0


def dep(ids):
    return ":".join(ids)


def main():
    parser = argparse.ArgumentParser(description="Submit the paper PCR rerun pipeline.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prepare_id = submit(PREPARE, dry_run=args.dry_run)
    baseline_ids = [
        submit(path, dependency=prepare_id, dry_run=args.dry_run)
        for path in BASELINES
    ]
    build_id = submit(BUILD, dependency=dep(baseline_ids), dry_run=args.dry_run)

    wga_ids = {{}}
    for key, path in WGA.items():
        wga_ids[key] = submit(path, dependency=build_id, dry_run=args.dry_run)

    for key, path in FACTONLY.items():
        submit(path, dependency=wga_ids[key], dry_run=args.dry_run)
    for key, path in CUEONLY.items():
        submit(path, dependency=wga_ids[key], dry_run=args.dry_run)

    print("Pipeline submitted.")
    print(f"prepare={{prepare_id}} build_global_eligible={{build_id}}")


if __name__ == "__main__":
    main()
'''
    helper_path = results_root / "submit_pipeline.py"
    helper_path.write_text(helper, encoding="utf-8")
    helper_path.chmod(0o755)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    sbatch_dir = results_root / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    prepare_path = sbatch_dir / "prepare_samples.sbatch"
    prepare_path.write_text(render_prepare_samples(args, results_root), encoding="utf-8")

    build_path = sbatch_dir / "build_global_eligible.sbatch"
    build_path.write_text(render_build_manifest(args, results_root), encoding="utf-8")

    baseline_paths = []
    attack_paths = {}

    for model_alias in args.models:
        for dataset in args.datasets:
            b_path = sbatch_dir / f"pcr_baseline_{model_alias}_{dataset}.sbatch"
            b_path.write_text(
                render_baseline(
                    args=args,
                    results_root=results_root,
                    model_alias=model_alias,
                    dataset=dataset,
                ),
                encoding="utf-8",
            )
            baseline_paths.append(b_path)

            for attack in args.attacks:
                path = sbatch_dir / f"pcr_{attack}_{model_alias}_{dataset}.sbatch"
                path.write_text(
                    render_attack(
                        args=args,
                        results_root=results_root,
                        model_alias=model_alias,
                        dataset=dataset,
                        attack=attack,
                    ),
                    encoding="utf-8",
                )
                attack_paths[(attack, model_alias, dataset)] = path

                log_dir = results_root / "logs" / model_alias / dataset
                out_dir = jsonl_path(results_root, model_alias, dataset, attack).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)

            baseline_path(results_root, model_alias, dataset).parent.mkdir(
                parents=True, exist_ok=True)

    for dataset in args.datasets:
        sample_path(results_root, dataset, args.n_sample).parent.mkdir(
            parents=True, exist_ok=True)
        eligible_path(results_root, dataset).parent.mkdir(parents=True, exist_ok=True)

    (results_root / "ddg_cache").mkdir(parents=True, exist_ok=True)
    (results_root / "logs" / "_pipeline").mkdir(parents=True, exist_ok=True)

    write_submit_order(
        results_root,
        prepare_path=prepare_path,
        build_path=build_path,
        baseline_paths=baseline_paths,
        attack_paths=attack_paths,
    )
    write_submit_helper(
        results_root,
        prepare_path=prepare_path,
        build_path=build_path,
        baseline_paths=baseline_paths,
        attack_paths=attack_paths,
    )

    print(f"Wrote {1 + 1 + len(baseline_paths) + len(attack_paths)} sbatch files under {sbatch_dir}")
    print(f"Wrote submit order to {results_root / 'submit_order.md'}")
    print(f"Wrote submit helper to {results_root / 'submit_pipeline.py'}")


if __name__ == "__main__":
    main()

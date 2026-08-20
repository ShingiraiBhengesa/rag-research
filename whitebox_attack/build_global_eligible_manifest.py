#!/usr/bin/env python3
"""
Build global clean-eligible sample manifests from paper baseline rollouts.

For each dataset, an example is retained only if every configured model has a
completed clean baseline row and that row is eligible. This gives one uniform
eligible sample set per dataset for all attacks and models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_utils import load_retrieval_control_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "whitebox_paper_results" / "pcr_full_completion_v1"
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")
MODELS = ("qwen7b", "qwen14b", "llama8b", "gemma27b", "gptoss20b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global eligible manifests.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--n-sample", type=int, default=450)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write manifests even if baseline files are missing rows.",
    )
    return parser.parse_args()


def baseline_path(results_root: Path, model: str, dataset: str) -> Path:
    return (
        results_root
        / "baselines"
        / model
        / dataset
        / f"clean_baseline_{model}_{dataset}.jsonl"
    )


def sample_path(results_root: Path, dataset: str, n_sample: int) -> Path:
    return results_root / "samples" / f"{dataset}_n{n_sample}.jsonl"


def load_clean_rows(path: Path, dataset: str) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if row.get("dataset") != dataset or row.get("condition") != "clean":
                continue
            rows[row["example_id"]] = row
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    out_dir = results_root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    had_incomplete = False
    summary_lines = [
        "# Global Eligible Samples",
        "",
        "An example is included only when every configured model's clean baseline",
        "row is present and eligible.",
        "",
        "| Dataset | Candidate Rows | Global Eligible | Missing Clean Rows | "
        + " | ".join(f"{model} Eligible" for model in args.models)
        + " |",
        "|---|---:|---:|---:|" + "|".join(["---:"] * len(args.models)) + "|",
    ]

    for dataset in args.datasets:
        s_path = sample_path(results_root, dataset, args.n_sample)
        if not s_path.exists():
            print(f"Missing sample manifest: {s_path}", file=sys.stderr)
            return_code = 2
            sys.exit(return_code)

        samples = load_retrieval_control_manifest(
            str(s_path), n_sample=args.n_sample, dataset_src=dataset)
        sample_ids = [row["name"] for row in samples]
        sample_by_id = {row["name"]: row for row in samples}

        clean_by_model = {}
        eligible_sets = {}
        missing_total = 0

        for model in args.models:
            rows = load_clean_rows(baseline_path(results_root, model, dataset), dataset)
            clean_by_model[model] = rows
            missing = [eid for eid in sample_ids if eid not in rows]
            missing_total += len(missing)
            if missing:
                had_incomplete = True
                print(
                    f"{dataset}/{model}: missing {len(missing)} clean rows "
                    f"(first missing: {missing[:3]})",
                    file=sys.stderr,
                )
            eligible_sets[model] = {
                eid for eid in sample_ids
                if rows.get(eid, {}).get("eligible")
            }

        global_eligible_ids = [
            eid for eid in sample_ids
            if all(eid in eligible_sets[model] for model in args.models)
        ]
        global_rows = [sample_by_id[eid] for eid in global_eligible_ids]
        out_path = out_dir / f"global_eligible_{dataset}.jsonl"
        write_jsonl(out_path, global_rows)

        eligible_counts = [len(eligible_sets[model]) for model in args.models]
        summary_lines.append(
            f"| {dataset} | {len(samples)} | {len(global_rows)} | {missing_total} | "
            + " | ".join(str(count) for count in eligible_counts)
            + " |"
        )
        print(
            f"{dataset}: candidates={len(samples)} "
            f"global_eligible={len(global_rows)} missing_clean_rows={missing_total} "
            f"wrote={out_path}",
            flush=True,
        )

    summary_path = out_dir / "global_eligible_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}", flush=True)

    if had_incomplete and not args.allow_incomplete:
        print(
            "Baseline rows are incomplete; manifests were written for inspection, "
            "but this job is failing so dependent attack jobs do not start.",
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()

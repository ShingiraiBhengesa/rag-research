#!/usr/bin/env python3
"""
Freeze canonical per-dataset sample manifests for paper reruns.

This records the exact examples produced by data_utils after the dataset-level
filters, so later clean baseline and attack jobs do not independently resample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_utils import load_retrieval_control_pool


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "whitebox_paper_results" / "pcr_full_completion_v1" / "samples"
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write canonical sample manifests.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-sample", type=int, default=450)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    return parser.parse_args()


def row_hash(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_lines = [
        "# Canonical Dataset Samples",
        "",
        f"`n_sample={args.n_sample}` per dataset.",
        "",
        "| Dataset | Rows | First ID | Last ID | Manifest | Hash |",
        "|---|---:|---|---|---|---|",
    ]

    for dataset in args.datasets:
        rows = load_retrieval_control_pool(
            n_sample=args.n_sample, dataset_src=dataset)
        path = out_dir / f"{dataset}_n{args.n_sample}.jsonl"
        write_jsonl(path, rows)
        all_rows.extend(rows)

        joined_hash = hashlib.sha256(
            "".join(row_hash(row) for row in rows).encode("utf-8")
        ).hexdigest()[:16]
        first_id = rows[0]["name"] if rows else ""
        last_id = rows[-1]["name"] if rows else ""
        summary_lines.append(
            f"| {dataset} | {len(rows)} | `{first_id}` | `{last_id}` | "
            f"`{path.name}` | `{joined_hash}` |"
        )
        print(f"{dataset}: wrote {len(rows)} rows to {path}")

    combined = out_dir / f"all_n{args.n_sample}.jsonl"
    write_jsonl(combined, all_rows)
    summary_lines.extend(["", f"Combined manifest: `{combined.name}`", ""])
    (out_dir / "sample_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8")
    print(f"combined: wrote {len(all_rows)} rows to {combined}")
    print(f"summary: {out_dir / 'sample_summary.md'}")


if __name__ == "__main__":
    main()

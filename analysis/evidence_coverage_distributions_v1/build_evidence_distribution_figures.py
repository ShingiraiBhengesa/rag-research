"""Top-level orchestrator for the evidence-coverage-distribution package.

Usage: python -m analysis.evidence_coverage_distributions_v1.build_evidence_distribution_figures [--stage data|figures|all]

Read-only against all canonical sources (analysis/evidence_hop_coverage/,
qwen_llama_results_bundle/, gemma_gptoss_results_bundle/, raw traces). Writes
only under analysis/evidence_coverage_distributions_v1/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import source_loader as sl
from . import categories as cat
from . import missing_evidence as me
from . import pairing as pr
from . import validation as val
from .writers import write_csv

OUT_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "coverage": OUT_DIR / "coverage_distribution_by_condition.csv",
    "missing": OUT_DIR / "missing_evidence_distribution.csv",
    "paired_loss": OUT_DIR / "paired_evidence_loss_distribution.csv",
    "transitions": OUT_DIR / "full_coverage_transitions.csv",
    "gates": OUT_DIR / "validation_gates.csv",
}


def run_stage_data() -> dict:
    rows = sl.load_all_rows()

    coverage_table = cat.build_coverage_distribution_table(rows, sl.MODEL_DATASETS, sl.ALL_CONDITIONS)
    write_csv(DATA_FILES["coverage"], coverage_table)

    missing_table = me.build_missing_evidence_table(rows, sl.MODEL_DATASETS, sl.ALL_CONDITIONS)
    write_csv(DATA_FILES["missing"], missing_table)

    paired_loss_table = pr.build_paired_loss_table(rows, sl.MODEL_DATASETS)
    write_csv(DATA_FILES["paired_loss"], paired_loss_table)

    transitions_table = pr.build_full_coverage_transitions_table(rows, sl.MODEL_DATASETS)
    write_csv(DATA_FILES["transitions"], transitions_table)

    gate_results = val.run_all_gates(rows, coverage_table, missing_table, paired_loss_table, transitions_table)
    write_csv(DATA_FILES["gates"], [
        {"gate_id": g.gate_id, "description": g.description, "status": g.status, "detail": g.detail}
        for g in gate_results
    ])
    n_fail = sum(1 for g in gate_results if g.status == "FAIL")
    print(f"Validation gates: {len(gate_results) - n_fail}/{len(gate_results)} passed")
    for g in gate_results:
        if g.status == "FAIL":
            print(f"  GATE {g.gate_id} FAILED: {g.description} -- {g.detail}")

    return {
        "rows": rows, "coverage_table": coverage_table, "missing_table": missing_table,
        "paired_loss_table": paired_loss_table, "transitions_table": transitions_table,
    }


def run_stage_figures(data: dict | None = None) -> None:
    from . import figures_slide, figures_ecdf, figures_missing, figures_paired
    if data is None:
        data = run_stage_data()
    rows = data["rows"]
    figures_slide.generate_all(rows)
    figures_ecdf.generate_all(rows)
    figures_missing.generate_all(rows, data["missing_table"])
    figures_paired.generate_all(rows, data["paired_loss_table"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "figures", "all"], default="all")
    args = parser.parse_args()

    if args.stage == "data":
        run_stage_data()
    elif args.stage == "figures":
        run_stage_figures()
    else:
        data = run_stage_data()
        run_stage_figures(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

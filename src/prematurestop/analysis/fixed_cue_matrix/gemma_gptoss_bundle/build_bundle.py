"""Top-level CLI orchestrator for `results/whitebox/e1_fixed_cue_matrix/gemma_gptoss/`.

    python -m prematurestop.analysis.fixed_cue_matrix.gemma_gptoss_bundle.build_bundle \\
        --bundle-root results/whitebox/e1_fixed_cue_matrix/gemma_gptoss --stage data

Sibling to `prematurestop.analysis.fixed_cue_matrix.results_bundle.build_bundle` (the Qwen/Llama
bundle's orchestrator) -- reuses that bundle's own writer functions directly
(pcr_full.py, paired_deltas.py, evidence_ext.py, process_metrics.py,
answer_ext.py, run_inventory.py, comparisons.py are all model-agnostic; see
this package's own README.md for the reuse audit that established this).
Only gate_1 (matrix completeness) and this orchestrator are bespoke, because
this bundle's model/dataset matrix is asymmetric (scope.py).

Deliberately NOT included in this first pass (may be added later, not
because they're unreusable -- they were simply out of scope for this pass):
retrieval-equivalence fingerprinting, the mechanism/trace-raster figures,
the .xlsx workbook, and LaTeX appendix tables.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prematurestop.evaluation import config
from prematurestop.evaluation.answer_evaluation import build_answer_records
from prematurestop.evaluation import completion as cc
from prematurestop.analysis.fixed_cue_matrix.results_bundle import answer_ext, evidence_ext, pcr_full, paired_deltas, process_metrics, run_inventory
from prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons import write_condition_comparison_manifest_csv
from prematurestop.evaluation.completion_inventory import _row_is_valid, build_inventory
from prematurestop.evaluation.evidence import build_evidence_records

from . import gates
from .scope import EXTENDED_BUNDLE_MODELS
from prematurestop.paths import RESULTS_E1_GEMMA_GPTOSS


def run_stage_data(bundle_root: Path, production_root: Path, max_workers: int) -> None:
    data_dir = bundle_root / "data"
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    print("[1/9] Building completed-shard inventory across all 4 datasets "
          "(gpt_oss_20b/2wiki will correctly report 0 shards -- see scope.py)...")
    # datasets left at its default (config.DATASETS, all 4) deliberately --
    # build_inventory's per-cell loop tolerates a model missing an entire
    # dataset (reports STATUS_NONE, no crash), and keeping all 4 datasets in
    # the inventory call is what makes that gap show up explicitly in
    # run_inventory.csv instead of silently not being checked at all.
    inventory = build_inventory(production_root, models=EXTENDED_BUNDLE_MODELS)
    print(f"       {len(inventory.rows)} rows across {len(inventory.cells)} cells")
    valid_rows = [r for r in inventory.rows if _row_is_valid(r)]
    print(f"       {len(valid_rows)} valid rows (final_status=ok, no error, trace_exists)")

    print("[2/9] Writing condition_comparison_manifest.csv...")
    write_condition_comparison_manifest_csv(data_dir / "condition_comparison_manifest.csv")

    print("[3/9] Building canonical run-level completion records (this is the expensive step)...")
    canonical_records = cc.build_canonical_records(inventory.rows, max_workers=max_workers)
    print(f"       {len(canonical_records)} canonical records built")

    print("[4/9] Writing pcr_run_level.csv, pcr_stepwise.csv, route_decomposition_stepwise.csv...")
    pcr_full.write_pcr_run_level_csv(canonical_records, data_dir / "pcr_run_level.csv")
    pcr_full.write_pcr_stepwise_full_csv(canonical_records, data_dir / "pcr_stepwise.csv")
    pcr_full.write_route_decomposition_stepwise_csv(canonical_records, data_dir / "route_decomposition_stepwise.csv")

    print("[5/9] Writing pcr_paired_deltas_stepwise.csv (bootstrap CIs -- can take a while)...")
    paired_deltas.write_pcr_paired_deltas_stepwise_csv(canonical_records, data_dir / "pcr_paired_deltas_stepwise.csv")

    print("[6/9] Building evidence records...")
    evidence_records = build_evidence_records(valid_rows, max_workers=max_workers)

    print("[7/9] Building process/recovery metrics...")
    recovery_by_key = process_metrics.build_recovery_by_key(canonical_records, evidence_records)
    process_metrics.write_process_metrics_csv(canonical_records, recovery_by_key, data_dir / "process_metrics.csv")

    print("[6/9] Writing evidence_run_level.csv, evidence_summary.csv...")
    evidence_ext.write_evidence_run_level_csv(evidence_records, recovery_by_key, data_dir / "evidence_run_level.csv")
    evidence_ext.write_evidence_summary_csv(canonical_records, evidence_records, data_dir / "evidence_summary.csv")

    print("[8/9] Building answer records...")
    answer_records = build_answer_records(valid_rows, production_root, max_workers=max_workers)

    print("[8/9] Writing answer_run_level.csv, answer_summary.csv, answer_paired_flips.csv...")
    answer_ext.write_answer_run_level_csv(answer_records, production_root, data_dir / "answer_run_level.csv")
    answer_ext.write_answer_summary_csv(answer_records, data_dir / "answer_summary.csv")
    answer_ext.write_answer_paired_flips_csv(answer_records, data_dir / "answer_paired_flips.csv")

    print("[2/9] Writing run_inventory.csv (needs eligibility counts from steps above; "
          "includes gpt_oss_20b/2wiki's 12 STATUS_NONE cells explicitly)...")
    run_inventory.write_run_inventory_csv(
        inventory.cells, canonical_records, evidence_records, answer_records,
        data_dir / "run_inventory.csv", generated_at,
    )

    print("[9/9] Running validation gates...")
    results = gates.run_all_gates(bundle_root)
    gates.write_gates_csv(results, data_dir / "VALIDATION_GATES.csv")
    all_passed = gates.print_gate_summary(results)
    print(f"\n{'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED -- see data/VALIDATION_GATES.csv'}")


def run_stage_figures(bundle_root: Path, max_workers: int) -> None:
    from . import figures_bundle

    print("Generating PCR attack-panel figures...")
    figures_bundle.generate_pcr_attack_panels(bundle_root / "data", bundle_root / "figures")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=RESULTS_E1_GEMMA_GPTOSS)
    parser.add_argument("--production-root", type=Path, default=config.DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--stage", choices=("data", "figures", "all"), default="data")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    args.bundle_root.mkdir(parents=True, exist_ok=True)
    (args.bundle_root / "data").mkdir(parents=True, exist_ok=True)

    if args.stage in ("data", "all"):
        run_stage_data(args.bundle_root, args.production_root, args.max_workers)
    if args.stage in ("figures", "all"):
        run_stage_figures(args.bundle_root, args.max_workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

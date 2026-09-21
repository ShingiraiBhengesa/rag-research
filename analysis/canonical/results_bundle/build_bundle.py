"""Top-level CLI orchestrator for `qwen_llama_results_bundle/`.

    python -m analysis.canonical.results_bundle.build_bundle \\
        --bundle-root qwen_llama_results_bundle --stage data

Mirrors `analysis/cli.py`'s `--stage` pattern. `--stage data` runs the full
steps 1-11 (freeze through gates) and is the primary, most expensive stage
(canonical/evidence/answer records are all built once and reused across every
downstream writer). `--stage figures`/`--stage tables`/`--stage workbook`/
`--stage docs` (added in a later pass) consume `data/*.csv` only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from analysis import config
from analysis.answer_evaluation import build_answer_records
from analysis.canonical import canonical_completion as cc
from analysis.completion_inventory import _row_is_valid, build_inventory
from analysis.evidence import build_evidence_records

from . import answer_ext, evidence_ext, freeze_manifest, gates, mechanism_ext, paired_deltas, pcr_full, process_metrics, retrieval_equivalence_ext, run_inventory
from .comparisons import COMPARISONS, write_condition_comparison_manifest_csv
from .scope import CANONICAL_BUNDLE_MODELS


def run_stage_data(bundle_root: Path, production_root: Path, max_workers: int) -> None:
    data_dir = bundle_root / "data"
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    print("[1/11] Building completed-shard inventory (this also fetches active-job status)...")
    # Explicit scope, not config.INCLUDED_MODELS -- see scope.py's docstring.
    # This bundle's committed output must not change if a third model
    # temporarily enters config.INCLUDED_MODELS for an unrelated reason.
    inventory = build_inventory(production_root, models=CANONICAL_BUNDLE_MODELS)
    print(f"       {len(inventory.rows)} rows across {len(inventory.cells)} cells")
    # canonical_completion/retrieval_equivalence guard per-row internally
    # (row.final_status != "ok" or not row.trace_exists -> a non-crashing
    # failure route/skip), so they take the full row set -- invalid rows
    # still need to be counted in PCR's n_total denominator. evidence.py and
    # answer_evaluation.py do NOT guard (they call load_trace_json/
    # parse_trace_file unconditionally and raise FileNotFoundError on a
    # missing trace), so they need pre-filtered rows -- same convention
    # analysis/cli.py uses (`valid_rows = [r for r in rows if
    # ci._row_is_valid(r)]`, cli.py:74).
    valid_rows = [r for r in inventory.rows if _row_is_valid(r)]
    print(f"       {len(valid_rows)} valid rows (final_status=ok, no error, trace_exists)")

    print("[1/11] Writing analysis_manifest.json...")
    freeze_manifest.write_analysis_manifest_json(inventory, config.REPO_ROOT, generated_at, bundle_root / "analysis_manifest.json")

    print("[2/11] Writing condition_comparison_manifest.csv...")
    write_condition_comparison_manifest_csv(data_dir / "condition_comparison_manifest.csv")

    print("[3/11] Building canonical run-level completion records (this is the expensive step)...")
    canonical_records = cc.build_canonical_records(inventory.rows, max_workers=max_workers)
    print(f"       {len(canonical_records)} canonical records built")

    print("[4/11] Writing pcr_run_level.csv, pcr_stepwise.csv, route_decomposition_stepwise.csv...")
    pcr_full.write_pcr_run_level_csv(canonical_records, data_dir / "pcr_run_level.csv")
    pcr_full.write_pcr_stepwise_full_csv(canonical_records, data_dir / "pcr_stepwise.csv")
    pcr_full.write_route_decomposition_stepwise_csv(canonical_records, data_dir / "route_decomposition_stepwise.csv")

    print("[5/11] Writing pcr_paired_deltas_stepwise.csv (bootstrap CIs -- can take a while)...")
    paired_deltas.write_pcr_paired_deltas_stepwise_csv(canonical_records, data_dir / "pcr_paired_deltas_stepwise.csv")

    print("[6/11] Building retrieval-equivalence fingerprints...")
    re_rows = retrieval_equivalence_ext.build_pair_comparison_rows_for_manifest(inventory.rows, max_workers=max_workers)
    retrieval_equivalence_ext.write_retrieval_equivalence_csv(re_rows, data_dir / "retrieval_equivalence.csv")

    print("[8/11] Building evidence records...")
    evidence_records = build_evidence_records(valid_rows, max_workers=max_workers)

    print("[7/11] Building process/recovery metrics...")
    recovery_by_key = process_metrics.build_recovery_by_key(canonical_records, evidence_records)
    process_metrics.write_process_metrics_csv(canonical_records, recovery_by_key, data_dir / "process_metrics.csv")

    print("[8/11] Writing evidence_run_level.csv, evidence_summary.csv...")
    evidence_ext.write_evidence_run_level_csv(evidence_records, recovery_by_key, data_dir / "evidence_run_level.csv")
    evidence_ext.write_evidence_summary_csv(canonical_records, evidence_records, data_dir / "evidence_summary.csv")

    print("[9/11] Building answer records...")
    answer_records = build_answer_records(valid_rows, production_root, max_workers=max_workers)

    print("[9-10/11] Writing answer_run_level.csv, answer_summary.csv, answer_paired_flips.csv...")
    answer_ext.write_answer_run_level_csv(answer_records, production_root, data_dir / "answer_run_level.csv")
    answer_ext.write_answer_summary_csv(answer_records, data_dir / "answer_summary.csv")
    answer_ext.write_answer_paired_flips_csv(answer_records, data_dir / "answer_paired_flips.csv")

    print("[10b/11] Writing mechanism_run_level.csv, mechanism_summary.csv (supports raster figures, not one of the 13 required data files)...")
    mechanism_ext.write_mechanism_csvs(
        canonical_records, evidence_records, COMPARISONS,
        data_dir / "mechanism_run_level.csv", data_dir / "mechanism_summary.csv",
    )

    print("[2/11] Writing run_inventory.csv (needs eligibility counts from steps above)...")
    run_inventory.write_run_inventory_csv(
        inventory.cells, canonical_records, evidence_records, answer_records,
        data_dir / "run_inventory.csv", generated_at,
    )

    print("[11/11] Running validation gates...")
    results = gates.run_all_gates(bundle_root)
    gates.write_gates_csv(results, data_dir / "VALIDATION_GATES.csv")
    all_passed = gates.print_gate_summary(results)
    print(f"\n{'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED -- see data/VALIDATION_GATES.csv'}")


def run_stage_figures(bundle_root: Path, production_root: Path, max_workers: int) -> None:
    from . import figures_bundle

    print("Generating figures...")
    figures_bundle.generate_all_figures(bundle_root, production_root, max_workers=max_workers)


def run_stage_tables(bundle_root: Path) -> None:
    from . import tables_bundle

    print("Generating Markdown tables...")
    tables_bundle.write_all_tables(bundle_root)


def run_stage_workbook(bundle_root: Path) -> None:
    from . import workbook

    print("Generating .xlsx workbook...")
    workbook.write_workbook(bundle_root / "data", bundle_root / "qwen_llama_results_workbook.xlsx")


def run_stage_docs(bundle_root: Path) -> None:
    from . import docs_bundle

    print("Generating README.md, MASTER_RESULTS.md, CHAPTER5_WRITING_NOTES.md, LAB_MEETING_BRIEF.md...")
    docs_bundle.write_all_docs(bundle_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path("qwen_llama_results_bundle"))
    parser.add_argument("--production-root", type=Path, default=config.DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--stage", choices=("data", "figures", "tables", "workbook", "docs", "all"), default="data")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    args.bundle_root.mkdir(parents=True, exist_ok=True)
    (args.bundle_root / "data").mkdir(parents=True, exist_ok=True)

    if args.stage in ("data", "all"):
        run_stage_data(args.bundle_root, args.production_root, args.max_workers)
    if args.stage in ("figures", "all"):
        run_stage_figures(args.bundle_root, args.production_root, args.max_workers)
    if args.stage in ("tables", "all"):
        run_stage_tables(args.bundle_root)
    if args.stage in ("workbook", "all"):
        run_stage_workbook(args.bundle_root)
    if args.stage in ("docs", "all"):
        run_stage_docs(args.bundle_root)
        # Gate 11 (every MASTER_RESULTS.md number is cited) can only be
        # checked once MASTER_RESULTS.md exists -- spec section 11 orders
        # "run all validation gates" (step 11) before "generate the
        # Markdown report" (step 12), so the data-stage gate run is
        # necessarily incomplete on gate 11 until docs exist. Re-run gates
        # here so `data/VALIDATION_GATES.csv` reflects the final state.
        print("Re-running validation gates now that MASTER_RESULTS.md exists...")
        results = gates.run_all_gates(args.bundle_root)
        gates.write_gates_csv(results, args.bundle_root / "data" / "VALIDATION_GATES.csv")
        all_passed = gates.print_gate_summary(results)
        print(f"\n{'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED -- see data/VALIDATION_GATES.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

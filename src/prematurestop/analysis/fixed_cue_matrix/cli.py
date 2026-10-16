"""Canonical CLI entry point for the completed Qwen/Llama analysis.

This is a legacy entry point (wires src/prematurestop/evaluation/pcr.py's deprecated
dual-denominator PCR implementation, not canonical_completion.py -- see
results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md). config.INCLUDED_MODELS
now includes all 4 models (gemma27b's fresh re-run and gpt_oss_20b's 3/4
completed datasets are both re-included; see config.MODEL_SCOPE_STATEMENT),
but this module's own --output-root default below still targets the
2-model qwen/llama directory it was built against -- pass an explicit
--output-root if running this legacy path against a different model scope.

Usage (from the repository root):

    python -m prematurestop.analysis.fixed_cue_matrix.cli \\
        --production-root results \\
        --output-root results/evaluation/v3_final_completed_qwen_llama_20260914 \\
        --stage all

Every path is CLI-configurable and repo-relative by default -- nothing in
this module hardcodes a machine-specific absolute path. `--stage` selects
one pipeline stage (`inventory`, `pcr`, `pairing`, `evidence`, `answers`,
`statistics`) or `all` to run them in dependency order.

This entry point only ever reads production files for `config.INCLUDED_MODELS`
(guarded by `config.assert_included_model` / `config.assert_path_not_excluded`
throughout the package) and writes exclusively under `--output-root`. It
never modifies anything under `results/traces`, `results/runs`, or any other
raw production directory, and never touches SLURM jobs beyond a read-only
`squeue` snapshot for the active-jobs audit.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from prematurestop.evaluation import (
    answer_evaluation as ae,
    completion_inventory as ci,
    config,
    evidence,
    pairing,
    pcr,
)
from prematurestop.stats import statistics as stats

STAGES = ("inventory", "pcr", "pairing", "evidence", "answers", "statistics")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--production-root", type=Path, default=config.DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional dir to cache intermediate pickles between stages")
    return parser.parse_args(argv)


def _cache_path(cache_dir: Path | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for model in config.INCLUDED_MODELS:
        config.assert_included_model(model)

    print(f"[cli] production_root={args.production_root}", file=sys.stderr)
    print(f"[cli] output_root={args.output_root}", file=sys.stderr)

    inventory_result = ci.build_inventory(args.production_root)
    valid_rows = [r for r in inventory_result.rows if ci._row_is_valid(r)]
    print(f"[cli] {len(valid_rows)} valid completed rows across {len(config.INCLUDED_MODELS)} models", file=sys.stderr)

    if args.stage in ("inventory", "all"):
        ci.write_inventory_csvs(inventory_result, args.output_root / "audits")
        print("[cli] wrote audits/", file=sys.stderr)

    pcr_records = None
    pcr_cache = _cache_path(args.cache_dir, "pcr_records.pkl")
    if args.stage in ("pcr", "statistics", "all"):
        if pcr_cache and pcr_cache.exists():
            pcr_records = pickle.loads(pcr_cache.read_bytes())
        else:
            pcr_records = pcr.build_pcr_records(valid_rows, max_workers=args.max_workers)
            if pcr_cache:
                pcr_cache.write_bytes(pickle.dumps(pcr_records))
        if args.stage in ("pcr", "all"):
            pcr.write_pcr_outputs(pcr_records, args.output_root)
            print("[cli] wrote pcr/ and by_model_dataset/", file=sys.stderr)

    pairing_audit = None
    if args.stage in ("pairing", "statistics", "all"):
        pairing_audit = pairing.build_pairing_audit(valid_rows, max_workers=args.max_workers)
        if args.stage in ("pairing", "all"):
            pairing.write_pairing_outputs(pairing_audit, args.output_root / "audits")
            print("[cli] wrote pairing audits/", file=sys.stderr)

    if args.stage in ("evidence", "all"):
        evidence_records = evidence.build_evidence_records(valid_rows, max_workers=args.max_workers)
        out_dir = args.output_root / "evidence"
        evidence.write_evidence_state_at_injection(evidence_records, out_dir / "evidence_state_at_injection.csv")
        evidence.write_evidence_coverage_by_boundary(evidence_records, out_dir / "evidence_coverage_by_boundary.csv")
        print("[cli] wrote evidence/", file=sys.stderr)

    if args.stage in ("answers", "all"):
        answer_records = ae.build_answer_records(valid_rows, args.production_root, max_workers=args.max_workers)
        out_dir = args.output_root / "answers"
        ae.write_predictions_jsonl(answer_records, out_dir / "final_answer_predictions.jsonl")
        ae.write_summary_csv(ae.summarize_by_cell(answer_records), out_dir / "final_answer_summary.csv")
        print("[cli] wrote answers/", file=sys.stderr)

    if args.stage in ("statistics", "all"):
        if pcr_records is None or pairing_audit is None:
            raise RuntimeError("statistics stage requires pcr_records and pairing_audit")
        records_by_key = {(r.model, r.dataset, r.condition, r.example_id): r for r in pcr_records}
        strictly_paired: dict[tuple, set[str]] = {}
        for row in pairing_audit["pairing"]:
            if row["strictly_paired"]:
                key = (row["model"], row["dataset"], row["condition_a"], row["condition_b"])
                strictly_paired.setdefault(key, set()).add(row["example_id"])
        results = stats.compute_paired_checkpoint_stats(records_by_key, strictly_paired)
        stats.write_paired_checkpoint_stats(results, args.output_root / "statistics" / "paired_checkpoint_stats.csv")
        print("[cli] wrote statistics/", file=sys.stderr)

    print("[cli] done", file=sys.stderr)


if __name__ == "__main__":
    run()

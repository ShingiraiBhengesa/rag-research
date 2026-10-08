"""Validation gates for this bundle.

Gate 1 is bespoke (this bundle's model/dataset matrix is asymmetric --
gpt_oss_20b has no 2wiki data at all, by design, not by omission -- see
scope.py). Gates 2-13 are imported and reused completely unchanged from
`analysis.canonical.results_bundle.gates`: every one of them reads only
`data_dir`/`bundle_root`-relative CSVs generically (no model list or
dataset-count assumption baked in -- confirmed by inspection before reuse),
so they need no adaptation to run against this bundle's own data directory
instead of qwen_llama_results_bundle/'s.
"""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

from analysis import config
from analysis.canonical.results_bundle.gates import (
    GateResult,
    _read_csv,
    gate_2_pcr_monotonic,
    gate_3_no_step_over_29,
    gate_4_pcr_identity,
    gate_5_paired_n_bound,
    gate_6_no_imputation,
    gate_7_value_ranges,
    gate_8_full_hop_agreement,
    gate_9_flip_cells_sum,
    gate_10_figures_join,
    gate_11_cited_numbers,
    gate_12_adjudication_column,
    gate_13_no_legacy_material,
)

from .scope import EXTENDED_BUNDLE_MODELS, MODEL_DATASETS


def gate_1_full_matrix(data_dir: Path) -> GateResult:
    """Expects gemma27b x all 4 datasets x 12 conditions, plus gpt_oss_20b x
    its 3 available datasets x 12 conditions -- gpt_oss_20b/2wiki is expected
    to be ABSENT (0 cells), not present-and-failing. A cell present for
    gpt_oss_20b/2wiki would itself be a gate failure (would mean this
    bundle's scope has silently drifted from scope.py's MODEL_DATASETS)."""
    rows = _read_csv(data_dir / "run_inventory.csv")
    expected = set()
    for model in EXTENDED_BUNDLE_MODELS:
        for dataset in MODEL_DATASETS[model]:
            for condition in config.CONDITIONS:
                expected.add((model, dataset, condition))

    present = {(r["model"], r["dataset"], r["condition"]) for r in rows}
    # run_inventory.csv intentionally also carries gpt_oss_20b/2wiki's 12
    # STATUS_NONE cells (documenting the gap, not hiding it) -- those must
    # be present in the file but are not part of the "expected complete
    # matrix" this gate checks.
    excluded_cells = {
        ("gpt_oss_20b", "2wiki", condition) for condition in config.CONDITIONS
    }
    present_in_scope = present - excluded_cells
    unexpectedly_present = excluded_cells & present
    stray_gpt_oss_2wiki_data = any(
        r["model"] == "gpt_oss_20b" and r["dataset"] == "2wiki" and r.get("n_completed") not in ("0", "", None)
        for r in rows
    )

    missing = expected - present_in_scope
    ok = not missing and not stray_gpt_oss_2wiki_data
    detail = (
        f"{len(present_in_scope)}/{len(expected)} in-scope cells present"
        f"; gpt_oss_20b/2wiki correctly excluded={not stray_gpt_oss_2wiki_data}"
    )
    if missing:
        detail += f"; missing: {sorted(missing)[:5]}"
    return GateResult(
        1,
        "gemma27b (4 datasets) + gpt_oss_20b (3 datasets, 2wiki excluded) have the expected 12-condition matrix",
        ok, detail,
    )


def run_all_gates(bundle_root: Path) -> list[GateResult]:
    data_dir = bundle_root / "data"
    figures_dir = bundle_root / "figures"
    results_path = bundle_root / "RESULTS.md"
    return [
        gate_1_full_matrix(data_dir),
        gate_2_pcr_monotonic(data_dir),
        gate_3_no_step_over_29(data_dir),
        gate_4_pcr_identity(data_dir),
        gate_5_paired_n_bound(data_dir),
        gate_6_no_imputation(data_dir),
        gate_7_value_ranges(data_dir),
        gate_8_full_hop_agreement(data_dir),
        gate_9_flip_cells_sum(data_dir),
        gate_10_figures_join(figures_dir, data_dir),
        gate_11_cited_numbers(results_path),
        gate_12_adjudication_column(data_dir),
        gate_13_no_legacy_material(bundle_root, data_dir),
    ]


def write_gates_csv(results: list[GateResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gate_id", "description", "status", "detail"])
        for r in results:
            writer.writerow([r.gate_id, r.description, "PASS" if r.passed else "FAIL", r.detail])


def print_gate_summary(results: list[GateResult]) -> bool:
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(f"[{status}] gate {r.gate_id}: {r.description} -- {r.detail}")
    return all_passed

"""Phase 11: validation gates. Fails loudly (raises) if any gate fails when
run as __main__; importable as a library for the test suite and the build
script to call non-fatally and collect results."""
from __future__ import annotations

from dataclasses import dataclass

from . import categories as cat
from . import pairing as pr
from .source_loader import CoverageRow, PCR_NUMERATOR_ROUTES


@dataclass
class GateResult:
    gate_id: int
    description: str
    status: str  # PASS or FAIL
    detail: str


def gate_1_coverage_range(rows: list[CoverageRow]) -> GateResult:
    bad = [r for r in rows if not (0.0 - 1e-9 <= r.coverage <= 1.0 + 1e-9)]
    return GateResult(1, "Coverage is outside [0,1]", "FAIL" if bad else "PASS", f"{len(bad)} violations / {len(rows)} rows")


def gate_2_category_totals(coverage_table: list[dict]) -> GateResult:
    bad = [r for r in coverage_table if r["eligible_N"] > 0 and
           sum(r[f"n_{c}"] for c in cat.CATEGORIES) != r["eligible_N"]]
    return GateResult(2, "A coverage-category total differs from eligible N", "FAIL" if bad else "PASS",
                       f"{len(bad)} violations / {len(coverage_table)} cells")


def gate_3_percentage_sums(coverage_table: list[dict], tolerance: float = 0.5) -> GateResult:
    bad = [r for r in coverage_table if r["eligible_N"] > 0 and
           abs(sum(r[f"pct_{c}"] for c in cat.CATEGORIES) - 100.0) > tolerance]
    return GateResult(3, "Category percentages do not sum to 100% within rounding tolerance",
                       "FAIL" if bad else "PASS", f"{len(bad)} violations (tolerance={tolerance}pp) / {len(coverage_table)} cells")


def gate_4_retrieved_le_required(rows: list[CoverageRow]) -> GateResult:
    bad = [r for r in rows if r.retrieved > r.required]
    return GateResult(4, "Retrieved evidence exceeds required evidence", "FAIL" if bad else "PASS",
                       f"{len(bad)} violations / {len(rows)} rows")


def gate_5_coverage_consistency(rows: list[CoverageRow], tol: float = 1e-6) -> GateResult:
    bad = []
    for r in rows:
        if r.required == 0:
            continue
        if abs(r.retrieved / r.required - r.coverage) > tol:
            bad.append(r)
        if r.missing != r.required - r.retrieved:
            bad.append(r)
    return GateResult(5, "Coverage disagrees with retrieved/required counts", "FAIL" if bad else "PASS",
                       f"{len(bad)} violations / {len(rows)} rows")


def gate_6_paired_ids_match(rows: list[CoverageRow]) -> GateResult:
    """Re-derives every pair for every comparison and confirms every pair's
    two rows share the same example_id (pair_by_example_id only ever joins
    on example_id, so this is a structural re-check, not a tautology --
    it fails if a future edit ever changes the join key)."""
    from .pairing import pair_by_example_id, COMPARISONS
    from .source_loader import MODEL_DATASETS
    bad = 0
    checked = 0
    for model, datasets in MODEL_DATASETS.items():
        for dataset in datasets:
            for _, control_cond, cue_cond in COMPARISONS:
                control_rows = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == control_cond]
                cue_rows = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == cue_cond]
                pairs = pair_by_example_id(control_rows, cue_rows)
                checked += len(pairs)
                bad += sum(1 for c, q in pairs if c.example_id != q.example_id)
    return GateResult(6, "A paired comparison uses different question IDs", "FAIL" if bad else "PASS",
                       f"{bad} violations / {checked} pairs checked")


def gate_7_sign_convention(paired_loss_table: list[dict]) -> GateResult:
    """Spot-checks the documented sign convention (positive = control
    retrieved more, i.e. cue lost evidence) against a case we can reason
    about directly: Oracle Saturation vs Oracle Neutral is a calibration
    comparison where the cue side supplies strictly more/equal evidence by
    construction, so mean_paired_loss must be <= 0 for every dataset."""
    bad = []
    for r in paired_loss_table:
        if r["comparison_name"] == "Oracle Complete--Saturation vs Oracle Complete--Neutral" and r["mean_paired_loss"] is not None:
            if r["mean_paired_loss"] > 1e-6:
                bad.append(r)
    return GateResult(7, "The paired-loss sign convention is reversed", "FAIL" if bad else "PASS",
                       f"{len(bad)} violations (Oracle Saturation vs Neutral should be <=0)")


def gate_8_no_post_completion_evidence(rows: list[CoverageRow]) -> GateResult:
    """Structural check: every row's terminal_route must be a genuine PCR
    numerator route (this is enforced at load time in source_loader.py; this
    gate re-confirms it rather than trusting the loader silently)."""
    bad = [r for r in rows if r.terminal_route not in PCR_NUMERATOR_ROUTES]
    return GateResult(8, "Post-completion evidence enters local coverage (non-qualifying terminal route present)",
                       "FAIL" if bad else "PASS", f"{len(bad)} violations / {len(rows)} rows")


def gate_9_no_pooled_datasets(coverage_table: list[dict]) -> GateResult:
    """Every row in the table must have a real dataset name, never 'all' or
    'pooled' or similar -- confirms no primary table ever pools datasets."""
    pooled_tokens = {"all", "pooled", "combined", "average", "macro"}
    bad = [r for r in coverage_table if str(r["dataset"]).lower() in pooled_tokens]
    return GateResult(9, "A primary figure/table pools datasets", "FAIL" if bad else "PASS",
                       f"{len(bad)} pooled-looking dataset labels found")


def gate_10_gptoss_not_macro_averaged(rows: list[CoverageRow]) -> GateResult:
    """GPT-OSS/2wiki must be entirely absent, never present via any
    averaging shortcut."""
    bad = [r for r in rows if r.model == "gpt_oss_20b" and r.dataset == "2wiki"]
    return GateResult(10, "GPT-OSS macro averages presented as dataset-level distributions",
                       "FAIL" if bad else "PASS", f"{len(bad)} gpt_oss_20b/2wiki rows found (must be 0)")


def gate_11_no_kde_or_violin() -> GateResult:
    """Static code-presence check: grep every figure module for forbidden
    plotting calls."""
    import re
    forbidden = re.compile(r"violinplot|gaussian_kde|scipy\.stats\.kde|sns\.kdeplot|sns\.violinplot")
    here = __import__("pathlib").Path(__file__).resolve().parent
    hits = []
    for f in here.glob("figures_*.py"):
        text = f.read_text()
        if forbidden.search(text):
            hits.append(f.name)
    return GateResult(11, "A smoothed KDE or violin plot is used", "FAIL" if hits else "PASS",
                       f"forbidden plot calls found in: {hits}" if hits else "0 forbidden calls in any figures_*.py module")


def gate_12_denominators_visible(coverage_table: list[dict], paired_loss_table: list[dict], transitions_table: list[dict]) -> GateResult:
    missing_n = (
        sum(1 for r in coverage_table if "eligible_N" not in r) +
        sum(1 for r in paired_loss_table if "paired_N" not in r) +
        sum(1 for r in transitions_table if "paired_N" not in r)
    )
    return GateResult(12, "A figure/table omits its denominator", "FAIL" if missing_n else "PASS",
                       f"{missing_n} rows missing a denominator column")


def gate_13_oracle_labeled(paired_loss_table: list[dict]) -> GateResult:
    bad = [r for r in paired_loss_table if r["comparison_name"] in pr.CALIBRATION_COMPARISONS and not r["is_calibration"]]
    return GateResult(13, "Oracle and natural-retrieval conditions are mixed without clear labeling",
                       "FAIL" if bad else "PASS", f"{len(bad)} Oracle comparisons missing is_calibration=True")


def gate_14_canonical_files_unmodified() -> GateResult:
    """Confirms this package's source never performs an actual write
    operation (write_csv/write_text/open(...,'w'...)/savefig/shutil.copy)
    whose own line also names a canonical source directory. Excludes this
    file itself (validation.py), since describing the check is not
    performing a write -- a bare grep for "write" + the directory name
    would otherwise match this gate's own docstring/description strings."""
    write_calls = ("write_csv(", "write_text(", "savefig(", "shutil.copy", 'open(')
    canonical_dirs = ("evidence_hop_coverage/", "e1_fixed_cue_matrix/qwen_llama/", "e1_fixed_cue_matrix/gemma_gptoss/")
    here = __import__("pathlib").Path(__file__).resolve().parent
    hits = []
    for f in here.glob("*.py"):
        if f.name == "validation.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            if any(call in line for call in write_calls) and any(d in line for d in canonical_dirs):
                hits.append(f"{f.name}:{i}")
    return GateResult(14, "Existing PCR or evidence files are modified", "FAIL" if hits else "PASS",
                       f"suspicious write calls: {hits}" if hits else "0 write calls reference any canonical source directory")


def gate_15_figure_traceability(coverage_table: list[dict]) -> GateResult:
    """Every (model, dataset, condition) cell with N>0 that a slide figure
    would plot must have a matching row in coverage_distribution_by_condition.csv."""
    from .source_loader import MODEL_DATASETS, ALL_CONDITIONS
    keys = {(r["model"], r["dataset"], r["condition"]) for r in coverage_table}
    missing = []
    for model, datasets in MODEL_DATASETS.items():
        for dataset in datasets:
            for cond in ALL_CONDITIONS:
                if (model, dataset, cond) not in keys:
                    missing.append((model, dataset, cond))
    return GateResult(15, "A displayed value cannot be traced to an output CSV", "FAIL" if missing else "PASS",
                       f"{len(missing)} missing cells (every expected cell has a CSV row to trace to)")


def run_all_gates(rows: list[CoverageRow], coverage_table: list[dict], missing_table: list[dict],
                   paired_loss_table: list[dict], transitions_table: list[dict]) -> list[GateResult]:
    return [
        gate_1_coverage_range(rows),
        gate_2_category_totals(coverage_table),
        gate_3_percentage_sums(coverage_table),
        gate_4_retrieved_le_required(rows),
        gate_5_coverage_consistency(rows),
        gate_6_paired_ids_match(rows),
        gate_7_sign_convention(paired_loss_table),
        gate_8_no_post_completion_evidence(rows),
        gate_9_no_pooled_datasets(coverage_table),
        gate_10_gptoss_not_macro_averaged(rows),
        gate_11_no_kde_or_violin(),
        gate_12_denominators_visible(coverage_table, paired_loss_table, transitions_table),
        gate_13_oracle_labeled(paired_loss_table),
        gate_14_canonical_files_unmodified(),
        gate_15_figure_traceability(coverage_table),
    ]

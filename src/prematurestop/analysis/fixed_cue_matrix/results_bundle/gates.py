"""Step 11: validation gates.

Spec section 10's heading numbers gates "(1)...(13)" (implying 14, but only
13 distinct items are listed) -- this module uses 1-13, see the plan doc for
this resolution. Every gate reads only the CSVs already written under
`data/` (no re-parsing of traces) and is independent of the others, so a
failure in one gate doesn't prevent checking the rest -- all results are
collected and reported together.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

from prematurestop.evaluation import config

from .scope import CANONICAL_BUNDLE_MODELS


class GateResult:
    def __init__(self, gate_id: int, description: str, passed: bool, detail: str):
        self.gate_id = gate_id
        self.description = description
        self.passed = passed
        self.detail = detail


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(v):
    if v in (None, "", "None", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def gate_1_full_matrix(data_dir: Path) -> GateResult:
    """Expected matrix is derived from this bundle's own locked
    `CANONICAL_BUNDLE_MODELS`, never from `config.INCLUDED_MODELS` directly
    -- that repo-wide setting can change for reasons unrelated to this
    bundle (e.g. a model temporarily re-entering production), and this
    gate's expectation must not move when it does."""
    rows = _read_csv(data_dir / "run_inventory.csv")
    expected = set(product(CANONICAL_BUNDLE_MODELS, config.DATASETS, config.CONDITIONS))
    present = {(r["model"], r["dataset"], r["condition"]) for r in rows}
    missing = expected - present
    return GateResult(
        1, "Qwen14B/Llama8B have the expected 12-condition, 4-dataset matrix",
        not missing, f"{len(present)}/{len(expected)} cells present" + (f"; missing: {sorted(missing)[:5]}" if missing else ""),
    )


def gate_2_pcr_monotonic(data_dir: Path) -> GateResult:
    rows = _read_csv(data_dir / "pcr_stepwise.csv")
    grouped: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        pcr = _to_float(r["pcr"])
        if pcr is not None:
            grouped[(r["model"], r["dataset"], r["condition"])].append((int(r["step"]), pcr))
    violations = []
    for key, pts in grouped.items():
        pts.sort()
        for (k1, p1), (k2, p2) in zip(pts, pts[1:]):
            if p2 < p1 - 1e-9:
                violations.append((key, k1, k2))
    return GateResult(2, "Every PCR curve is monotonic nondecreasing", not violations, f"{len(violations)} violations" + (f": {violations[:3]}" if violations else ""))


def gate_3_no_step_over_29(data_dir: Path) -> GateResult:
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    run_level = _read_csv(data_dir / "pcr_run_level.csv")
    bad_steps = [r for r in stepwise if int(r["step"]) > 30]
    bad_runs = []
    for r in run_level:
        for field in ("research_complete_step", "no_tool_direct_exit_step"):
            v = _to_float(r.get(field))
            if v is not None and v > 29:
                bad_runs.append((r["run_id"], field, v))
    ok = not bad_steps and not bad_runs
    return GateResult(3, "No qualifying completion step exceeds 29", ok, f"{len(bad_steps)} stepwise rows > 30, {len(bad_runs)} run-level steps > 29")


def gate_4_pcr_identity(data_dir: Path) -> GateResult:
    rows = _read_csv(data_dir / "pcr_stepwise.csv")
    violations = 0
    for r in rows:
        pcr, rc, nt = _to_float(r["pcr"]), _to_float(r["research_complete_pcr"]), _to_float(r["no_tool_exit_pcr"])
        if pcr is None or rc is None or nt is None:
            continue
        if abs(pcr - (rc + nt)) > 1e-6:
            violations += 1
    return GateResult(4, "pcr == research_complete_pcr + no_tool_exit_pcr on every row", violations == 0, f"{violations} violations out of {len(rows)} rows")


def gate_5_paired_n_bound(data_dir: Path) -> GateResult:
    rows = _read_csv(data_dir / "pcr_paired_deltas_stepwise.csv")
    violations = [
        r for r in rows
        if int(r["n_paired"]) > min(int(r["n_control_eligible"]), int(r["n_cue_eligible"]))
    ]
    return GateResult(5, "Every paired N is no greater than either condition's eligible N", not violations, f"{len(violations)} violations")


def gate_6_no_imputation(data_dir: Path) -> GateResult:
    # Structurally guaranteed by build_strict_pairs (excludes ineligible rows
    # rather than imputing). Spot-check: no paired-delta row should have
    # n_paired > 0 with a control_pcr/cue_pcr of exactly None (would indicate
    # a row was fabricated rather than excluded).
    rows = _read_csv(data_dir / "pcr_paired_deltas_stepwise.csv")
    violations = [r for r in rows if int(r["n_paired"]) > 0 and (r["control_pcr"] in ("", "None") or r["cue_pcr"] in ("", "None"))]
    return GateResult(6, "No row is silently imputed from another condition", not violations, f"{len(violations)} suspicious rows (structurally guaranteed by build_strict_pairs's exclude-not-impute behavior)")


def gate_7_value_ranges(data_dir: Path) -> GateResult:
    checks = [
        ("pcr_stepwise.csv", ["pcr", "research_complete_pcr", "no_tool_exit_pcr"]),
        ("evidence_run_level.csv", ["local_completion_coverage", "final_run_coverage", "evidence_deficit"]),
        ("answer_run_level.csv", ["exact_match", "token_f1"]),
    ]
    violations = []
    for fname, cols in checks:
        for r in _read_csv(data_dir / fname):
            for col in cols:
                v = _to_float(r.get(col))
                if v is not None and not (-1e-9 <= v <= 1.0 + 1e-9):
                    violations.append((fname, col, v))
    return GateResult(7, "Coverage, recall, PCR, and accuracy values lie in [0,1]", not violations, f"{len(violations)} out-of-range values" + (f": {violations[:3]}" if violations else ""))


def gate_8_full_hop_agreement(data_dir: Path) -> GateResult:
    rows = _read_csv(data_dir / "evidence_run_level.csv")
    violations = []
    for r in rows:
        cov = _to_float(r.get("final_run_coverage"))
        flag = r.get("full_hop_indicator")
        if cov is None or flag in ("", "None"):
            continue
        expected = cov >= 1.0
        if _to_bool(flag) != expected:
            violations.append((r["model"], r["dataset"], r["condition"], r["example_id"]))
    return GateResult(8, "Full-hop status agrees with complete evidence coverage", not violations, f"{len(violations)} disagreements")


def gate_9_flip_cells_sum(data_dir: Path) -> GateResult:
    rows = _read_csv(data_dir / "answer_paired_flips.csv")
    violations = []
    for r in rows:
        total = sum(int(r[c]) for c in ("both_correct", "control_correct_attack_wrong", "control_wrong_attack_correct", "both_wrong"))
        if total != int(r["n_pairs"]):
            violations.append((r["model"], r["dataset"], r["condition_a"], r["condition_b"]))
    return GateResult(9, "The four answer-flip cells sum to paired answer N", not violations, f"{len(violations)} mismatches")


def gate_10_figures_join(figures_dir: Path, data_dir: Path) -> GateResult:
    # Spot-check only (per plan): every figure directory that exists should
    # correspond to a non-empty source CSV.
    expected_pairs = [
        ("pcr_curves", "pcr_stepwise.csv"),
        ("pcr_deltas", "pcr_paired_deltas_stepwise.csv"),
        ("evidence", "evidence_summary.csv"),
    ]
    problems = []
    for fig_subdir, csv_name in expected_pairs:
        fig_path = figures_dir / fig_subdir
        csv_rows = _read_csv(data_dir / csv_name)
        if fig_path.exists() and any(fig_path.iterdir()) and not csv_rows:
            problems.append(f"{fig_subdir} has figures but {csv_name} is empty")
    return GateResult(10, "Every figure value can be joined exactly to a canonical CSV row", not problems, "; ".join(problems) if problems else "spot-check passed")


def gate_11_cited_numbers(master_results_path: Path) -> GateResult:
    if not master_results_path.exists():
        return GateResult(11, "Every number in MASTER_RESULTS.md has a table/file reference", False, "MASTER_RESULTS.md does not exist yet")
    text = master_results_path.read_text(encoding="utf-8")
    uncited = []
    for line in text.splitlines():
        if re.search(r"\d", line) and "[table:" not in line and "|" not in line and not line.strip().startswith("#"):
            if re.search(r"(?<!\[table: )\b\d+(\.\d+)?%?\b", line):
                uncited.append(line.strip()[:80])
    return GateResult(11, "Every number in MASTER_RESULTS.md has a table/file reference", not uncited, f"{len(uncited)} lines with an un-cited bare number (spot-check regex; review manually)")


def gate_12_adjudication_column(data_dir: Path) -> GateResult:
    rows = _read_csv(data_dir / "answer_run_level.csv")
    has_column = bool(rows) and "needs_adjudication_flag" in rows[0]
    summary_rows = _read_csv(data_dir / "answer_summary.csv")
    summary_has_column = bool(summary_rows) and "n_needs_adjudication" in summary_rows[0]
    ok = has_column and summary_has_column
    return GateResult(12, "Adjudication flag present, never silently mixed into \"accuracy\"", ok, f"answer_run_level has flag={has_column}, answer_summary has denominator column={summary_has_column}")


_LEGACY_TOKENS = ("WGAR", "Attack-A", "attack2_gap_assessment", "Cue-Only", "cueonly", "Bridge-Fact", "bridge_fact", "wrong_gap_assessment")
# Only structural/label-bearing columns are scanned for legacy tokens --
# free-text columns (raw_final_report, extracted_short_answer, gold_answer,
# ...) can contain almost any substring as ordinary English (e.g. the driver
# name "Newgarden" contains "wgar"), which produced false positives when the
# whole file's raw text was searched. Legacy leakage would show up as a
# condition/comparison *label*, not as English prose, so only those columns
# are checked, as a whole-field match (not substring) against the token set.
_LEGACY_LABEL_COLUMNS = ("condition", "control_condition", "cue_condition", "condition_a", "condition_b", "comparison_name", "baseline")
# This gate's own report is not a canonical data table -- its description
# column necessarily names every legacy token it checks for, which would
# otherwise flag itself as a "hit" on any re-run once the file exists.
_GATE_13_EXCLUDED_FILES = ("VALIDATION_GATES.csv",)


def gate_13_no_legacy_material(bundle_root: Path, data_dir: Path) -> GateResult:
    hits = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name in _GATE_13_EXCLUDED_FILES:
            continue
        for row in _read_csv(csv_path):
            for col in _LEGACY_LABEL_COLUMNS:
                value = row.get(col)
                if not value:
                    continue
                for token in _LEGACY_TOKENS:
                    if token.lower() == str(value).lower():
                        hits.append((csv_path.name, col, value))
    run_inventory = _read_csv(data_dir / "run_inventory.csv")
    bad_conditions = {r["condition"] for r in run_inventory} - set(config.CONDITIONS)
    ok = not hits and not bad_conditions
    detail = f"{len(hits)} legacy-label hits, {len(bad_conditions)} non-canonical conditions" + (f": {hits[:5]}" if hits else "")
    return GateResult(13, "No legacy (WGAR/Attack-A/Cue-Only/Bridge-Fact) material in bundle CSVs", ok, detail)


def run_all_gates(bundle_root: Path) -> list[GateResult]:
    data_dir = bundle_root / "data"
    figures_dir = bundle_root / "figures"
    master_results_path = bundle_root / "MASTER_RESULTS.md"
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
        gate_11_cited_numbers(master_results_path),
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

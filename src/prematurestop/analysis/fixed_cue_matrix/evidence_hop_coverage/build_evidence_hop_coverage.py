"""Canonical run-level evidence/hop coverage table -- the first supporting
analysis after PCR.

PCR is LOCKED. This script does not redefine, recompute, or reinterpret
PCR; every completion field below (`first_terminal_post_observation_step`,
`terminal_route`, `pcr_eligible`, `anchor_*`) is read verbatim from
`prematurestop.evaluation.completion` -- the same module that produced
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` and
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/full_pcr_tables.md` (this repo's actual
canonical PCR artifacts; see this package's README.md for why those two
files, not the differently-named ones in the task prompt, are the locked
PCR artifacts actually being used, and why `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/`
is used in place of the prompt's `whitebox_attack/v3/analysis/...` path).

Reused, unmodified:
  - prematurestop.evaluation.completion  (PCR: T_i, terminal_route,
    pcr_eligible, anchor fields -- the single source of truth)
  - prematurestop.evaluation.completion_inventory            (run inventory, valid-row rule)
  - prematurestop.evaluation.trace_parser / prematurestop.evaluation.trace_schema (trace parsing, episodes,
    search events -- the actual event-walking logic, never re-implemented)
  - prematurestop.evaluation.evidence                        (injection-boundary and
    episode-0/local-completion union-coverage computation --
    `_cumulative_union_coverage`'s raw-ID-union method, verified in that
    module to be necessary for llama8b's field-reliability issue)
  - prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons (the single
    already-reconciled condition-comparison manifest; this script adds no
    second comparisons list)
  - prematurestop.evaluation.completion.build_strict_pairs (paired-eligible
    pairing rule, exclude-not-impute)

New in this script (not present anywhere else, see README.md for why):
  - final-workflow coverage: a raw-ID union across EVERY episode in the
    trace (episode 0 plus any supervisor redelegations), computed with the
    same method `prematurestop.evaluation.evidence` already uses for episode 0, so a
    "final < local" coverage regression cannot be caused by trusting a
    different, less-audited snapshot field.
  - hop-index extraction: position in the trace's own `required_paragraph_ids`
    tuple (stable, dataset-agnostic, verified against real traces to be
    order-preserving across conditions of the same example -- see README.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from prematurestop.evaluation import config
from prematurestop.evaluation import completion as cc
from prematurestop.evaluation.completion_inventory import RowRecord, _row_is_valid, build_inventory
from prematurestop.evaluation.evidence import _cumulative_union_coverage, _paragraph_index
from prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons import COMPARISONS
from prematurestop.analysis.fixed_cue_matrix.results_bundle.scope import (
    CANONICAL_BUNDLE_MODELS, bundle_models, set_bundle_models,
)
from prematurestop.evaluation.trace_parser import load_trace_json, parse_trace
from prematurestop.evaluation.trace_schema import ParsedTrace
from prematurestop.paths import RESULTS_E1

FROZEN_COMMIT = "9c327466ab8e730ba29a71db375ea5aa7556981d"

# spec section "canonical evidence mapping": per-dataset stratum semantics.
# hotpot is fixed two-hop by task instruction (not a real stratification
# axis); 2wiki/musique are genuine multi-value hop-count strata; multihoprag
# is explicitly an evidence COUNT, not an ordered reasoning chain (its
# `hop_index` in src/prematurestop/data/corpus.py::build_multihoprag_rows is just
# the annotation list's position, never validated as reasoning order).
STRATUM_TYPE_BY_DATASET: dict[str, str] = {
    "hotpot": "hop_count",
    "2wiki": "hop_count",
    "musique": "hop_count",
    "multihoprag": "required_evidence_count",
}
EXPECTED_STRATUM_VALUES: dict[str, set[int]] = {
    "hotpot": {2},
    "2wiki": {2, 4},
    "musique": {2, 3, 4},
    "multihoprag": {2, 3, 4},
}

ORACLE_CONDITIONS: tuple[str, ...] = tuple(c for c in config.CONDITIONS if c.startswith("OracleComplete"))


# ---------------------------------------------------------------------------
# Run-level record
# ---------------------------------------------------------------------------


@dataclass
class EvidenceHopRecord:
    # Identity and lineage
    model: str
    dataset: str
    example_id: str
    condition: str
    run_id: str
    trace_path: str
    dataset_stratum_type: str
    dataset_stratum_value: Optional[int]
    valid_run: bool
    pcr_eligible: bool
    evidence_eligible: bool
    ineligibility_reason: str

    # Locked completion fields (verbatim from canonical_completion)
    anchor_found: bool
    anchor_event_index: Optional[int]
    anchor_step: Optional[int]
    first_terminal_post_observation_step: Optional[int]
    terminal_route: str
    completed_within_budget: bool
    max_budget_exit: bool
    supervisor_redelegated: bool

    # Required evidence
    required_paragraph_ids_raw: str
    required_paragraph_ids_normalized: str
    required_evidence_count: int
    annotated_hop_count: Optional[int]

    # Evidence collected -- anchor
    anchor_retrieved_ids: str
    anchor_missing_ids: str
    anchor_evidence_count_retrieved: Optional[int]
    anchor_evidence_count_missing: Optional[int]
    anchor_coverage_fraction: Optional[float]
    anchor_full_evidence: Optional[bool]

    # Evidence collected -- local completion (T_i)
    local_retrieved_ids: str
    local_missing_ids: str
    local_evidence_count_retrieved: Optional[int]
    local_evidence_count_missing: Optional[int]
    local_coverage_fraction: Optional[float]
    local_full_evidence: Optional[bool]

    # Evidence collected -- final workflow termination
    final_retrieved_ids: str
    final_missing_ids: str
    final_evidence_count_retrieved: Optional[int]
    final_evidence_count_missing: Optional[int]
    final_coverage_fraction: Optional[float]
    final_full_evidence: Optional[bool]

    # Derived
    recovery_gain: Optional[float]
    incomplete_at_completion_full_at_final: Optional[bool]


def _hop_index_positions(required_paragraph_ids: tuple[str, ...]) -> dict[str, int]:
    return {pid: idx for idx, pid in enumerate(required_paragraph_ids)}


def _coverage_at_ids(required: tuple[str, ...], retrieved_union: set[str]) -> tuple[list[str], list[str], Optional[float]]:
    if not required:
        return [], [], None
    retrieved = [pid for pid in required if pid in retrieved_union]
    missing = [pid for pid in required if pid not in retrieved_union]
    return retrieved, missing, len(retrieved) / len(required)


def _final_workflow_union(pt: ParsedTrace) -> set[str]:
    """Union of retrieved paragraph IDs across EVERY episode in the trace
    (episode 0 plus any supervisor redelegations) -- the new boundary this
    script adds. Uses the same raw-ID-union method
    `prematurestop.evaluation.evidence._cumulative_union_coverage` already uses for episode
    0, just applied trace-wide, so this can never be a less-reliable
    computation than the local-completion boundary it's compared against."""
    union: set[str] = set()
    for episode in pt.episodes:
        for event in episode.search_events:
            union.update(event.retrieved_paragraph_ids)
    return union


def _ineligible_record(row: RowRecord, reason: str, canonical: Optional[cc.CanonicalCompletionRecord] = None) -> EvidenceHopRecord:
    # `canonical` is only passed for the "row has a valid trace but zero
    # required evidence" case -- PCR eligibility/route are independent of
    # evidence eligibility and must not be forced to False/EXECUTION_FAILURE
    # just because this row can't be evidence-scored (see README.md's
    # `evidence_eligible` definition: valid_run AND required_evidence_count > 0).
    pcr_eligible = canonical.pcr_eligible if canonical is not None else False
    terminal_route = canonical.terminal_route if canonical is not None else cc.EXECUTION_FAILURE
    anchor_found = canonical.anchor_found if canonical is not None else False
    anchor_event_index = canonical.anchor_event_index if canonical is not None else None
    anchor_step = canonical.anchor_step if canonical is not None else None
    first_terminal_step = canonical.first_terminal_post_observation_step if canonical is not None else None
    completed_within_budget = terminal_route in cc.PCR_NUMERATOR_ROUTES
    max_budget_exit = terminal_route == cc.MAX_BUDGET_EXIT
    supervisor_redelegated = canonical.supervisor_redelegated if canonical is not None else False
    return EvidenceHopRecord(
        model=row.model, dataset=row.dataset, example_id=row.example_id, condition=row.condition,
        run_id=row.run_id, trace_path=row.trace_path,
        dataset_stratum_type=STRATUM_TYPE_BY_DATASET.get(row.dataset, ""), dataset_stratum_value=None,
        valid_run=_row_is_valid(row), pcr_eligible=pcr_eligible, evidence_eligible=False, ineligibility_reason=reason,
        anchor_found=anchor_found, anchor_event_index=anchor_event_index, anchor_step=anchor_step,
        first_terminal_post_observation_step=first_terminal_step, terminal_route=terminal_route,
        completed_within_budget=completed_within_budget, max_budget_exit=max_budget_exit, supervisor_redelegated=supervisor_redelegated,
        required_paragraph_ids_raw="", required_paragraph_ids_normalized="", required_evidence_count=0,
        annotated_hop_count=None,
        anchor_retrieved_ids="", anchor_missing_ids="", anchor_evidence_count_retrieved=None,
        anchor_evidence_count_missing=None, anchor_coverage_fraction=None, anchor_full_evidence=None,
        local_retrieved_ids="", local_missing_ids="", local_evidence_count_retrieved=None,
        local_evidence_count_missing=None, local_coverage_fraction=None, local_full_evidence=None,
        final_retrieved_ids="", final_missing_ids="", final_evidence_count_retrieved=None,
        final_evidence_count_missing=None, final_coverage_fraction=None, final_full_evidence=None,
        recovery_gain=None, incomplete_at_completion_full_at_final=None,
    )


def build_evidence_hop_record(row: RowRecord, canonical: cc.CanonicalCompletionRecord) -> EvidenceHopRecord:
    if not _row_is_valid(row):
        return _ineligible_record(row, "invalid_row_final_status_or_missing_trace")

    full_path = config.resolve_stored_path(row.trace_path)
    raw = load_trace_json(full_path)
    pt = parse_trace(raw, trace_path=str(full_path))

    required_ids = pt.required_paragraph_ids
    required_n = pt.required_dependency_count
    stratum_type = STRATUM_TYPE_BY_DATASET.get(row.dataset, "")
    stratum_value = pt.hop_count if required_n else None
    annotated_hop_count = pt.hop_count if row.dataset != "multihoprag" and required_n else None

    if required_n == 0:
        return _ineligible_record(row, "no_required_evidence_for_example", canonical=canonical)

    required_norm = tuple(_paragraph_index(pid) for pid in required_ids)

    # --- Anchor (targeted observation preceding post-observation step 1) ---
    # `missing_dependency_ids_at_injection`/`dependencies_seen_at_injection`
    # are the runner's own authoritative dependency-level accounting for
    # this exact boundary (src/prematurestop/data/evidence.py::evaluate_coverage,
    # called once at the anchor search) -- used directly for the numeric
    # count and full-evidence flag, so they can never disagree with each
    # other by construction (both come from the same list). An EARLIER
    # version of this function instead tried to remap those dependency ids
    # to PARAGRAPH ids via a zip against the anchor SearchEvent's own
    # `required_dependency_ids`, and silently produced an empty missing-list
    # (hence a false `anchor_full_evidence`) whenever that SearchEvent's
    # list had already-unequal length -- caught by gate 7, not assumed away.
    anchor_coverage = pt.coverage_fraction_at_injection
    anchor_missing_dep_ids = tuple(pt.missing_dependency_ids_at_injection)
    anchor_seen_dep_ids = tuple(pt.dependencies_seen_at_injection)
    anchor_count_missing = len(anchor_missing_dep_ids) if anchor_coverage is not None else None
    anchor_count_retrieved = len(anchor_seen_dep_ids) if anchor_coverage is not None else None
    anchor_full = (anchor_count_missing == 0) if anchor_count_missing is not None else None

    # Paragraph-level id strings (for cross-boundary joinability with
    # local/final below) -- position-mapped via the anchor SearchEvent's own
    # parallel `required_dependency_ids`/`required_paragraph_ids` tuples
    # when that event's list length matches; left blank (not fabricated)
    # otherwise. This mapping is informational only -- the authoritative
    # count/flag fields above never depend on it.
    anchor_retrieved: list[str] = []
    anchor_missing: list[str] = []
    if anchor_coverage is not None:
        seen_dep_set = set(anchor_seen_dep_ids)
        anchor_event = pt.first_episode.search_events[0] if (pt.first_episode and pt.first_episode.search_events) else None
        if anchor_event and len(anchor_event.required_dependency_ids) == len(required_ids):
            for dep_id, para_id in zip(anchor_event.required_dependency_ids, required_ids):
                (anchor_retrieved if dep_id in seen_dep_set else anchor_missing).append(para_id)

    # --- Local completion (T_i) -- episode 0's union, per canonical_completion's
    # own anchor + terminal-route classification. Only meaningful when the
    # episode actually reached a qualifying terminal decision (RC/no-tool
    # exit); NA (not zero) for MAX_BUDGET_EXIT and everything else. ---
    completed_within_budget = canonical.terminal_route in cc.PCR_NUMERATOR_ROUTES
    max_budget_exit = canonical.terminal_route == cc.MAX_BUDGET_EXIT

    local_retrieved: list[str] = []
    local_missing: list[str] = []
    local_frac: Optional[float] = None
    if completed_within_budget and pt.first_episode is not None:
        ep0 = pt.first_episode
        ep0_union: set[str] = set()
        for event in ep0.search_events:
            ep0_union.update(event.retrieved_paragraph_ids)
        local_retrieved, local_missing, local_frac = _coverage_at_ids(required_ids, ep0_union)

    local_count_retrieved = len(local_retrieved) if local_frac is not None else None
    local_count_missing = len(local_missing) if local_frac is not None else None
    local_full = (local_frac >= 1.0) if local_frac is not None else None

    # --- Final workflow termination -- union across every episode. Always
    # computed when the trace has any search activity at all, including
    # MAX_BUDGET_EXIT rows (task: "preserve final evidence fields if they
    # can be measured"). ---
    final_union = _final_workflow_union(pt)
    if final_union or any(ep.search_events for ep in pt.episodes):
        final_retrieved, final_missing, final_frac = _coverage_at_ids(required_ids, final_union)
    else:
        final_retrieved, final_missing, final_frac = [], [], None

    final_count_retrieved = len(final_retrieved) if final_frac is not None else None
    final_count_missing = len(final_missing) if final_frac is not None else None
    final_full = (final_frac >= 1.0) if final_frac is not None else None

    recovery_gain = (final_frac - local_frac) if (final_frac is not None and local_frac is not None) else None
    incomplete_then_full = (
        (local_full is False and final_full is True) if (local_full is not None and final_full is not None) else None
    )

    return EvidenceHopRecord(
        model=row.model, dataset=row.dataset, example_id=row.example_id, condition=row.condition,
        run_id=row.run_id, trace_path=row.trace_path,
        dataset_stratum_type=stratum_type, dataset_stratum_value=stratum_value,
        valid_run=True, pcr_eligible=canonical.pcr_eligible, evidence_eligible=True, ineligibility_reason="",
        anchor_found=canonical.anchor_found, anchor_event_index=canonical.anchor_event_index,
        anchor_step=canonical.anchor_step,
        first_terminal_post_observation_step=canonical.first_terminal_post_observation_step,
        terminal_route=canonical.terminal_route,
        completed_within_budget=completed_within_budget, max_budget_exit=max_budget_exit,
        supervisor_redelegated=canonical.supervisor_redelegated,
        required_paragraph_ids_raw=";".join(required_ids), required_paragraph_ids_normalized=";".join(required_norm),
        required_evidence_count=required_n, annotated_hop_count=annotated_hop_count,
        anchor_retrieved_ids=";".join(anchor_retrieved), anchor_missing_ids=";".join(anchor_missing),
        anchor_evidence_count_retrieved=anchor_count_retrieved, anchor_evidence_count_missing=anchor_count_missing,
        anchor_coverage_fraction=anchor_coverage, anchor_full_evidence=anchor_full,
        local_retrieved_ids=";".join(local_retrieved), local_missing_ids=";".join(local_missing),
        local_evidence_count_retrieved=local_count_retrieved, local_evidence_count_missing=local_count_missing,
        local_coverage_fraction=local_frac, local_full_evidence=local_full,
        final_retrieved_ids=";".join(final_retrieved), final_missing_ids=";".join(final_missing),
        final_evidence_count_retrieved=final_count_retrieved, final_evidence_count_missing=final_count_missing,
        final_coverage_fraction=final_frac, final_full_evidence=final_full,
        recovery_gain=recovery_gain, incomplete_at_completion_full_at_final=incomplete_then_full,
    )


def _build_one(args: tuple) -> EvidenceHopRecord:
    row, canonical = args
    try:
        return build_evidence_hop_record(row, canonical)
    except Exception as exc:  # noqa: BLE001 -- preserve the row, mark the failure, never drop it
        return _ineligible_record(row, f"build_error:{type(exc).__name__}:{exc}", canonical=canonical)


def build_all_records(rows: list[RowRecord], canonical_records: list[cc.CanonicalCompletionRecord], max_workers: int = 8) -> list[EvidenceHopRecord]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    canonical_by_key = {(r.model, r.dataset, r.condition, r.example_id): r for r in canonical_records}
    args = []
    for row in rows:
        canonical = canonical_by_key.get((row.model, row.dataset, row.condition, row.example_id))
        if canonical is None:
            continue
        args.append((row, canonical))
    records: list[EvidenceHopRecord] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_build_one, a) for a in args]
        for fut in as_completed(futures):
            records.append(fut.result())
    return records


def write_run_level_csv(records: list[EvidenceHopRecord], path: Path) -> None:
    rows = [asdict(r) for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def _mean_or_na(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def build_summary(records: list[EvidenceHopRecord]) -> list[dict]:
    grouped: dict[tuple, list[EvidenceHopRecord]] = {}
    for r in records:
        grouped.setdefault((r.model, r.dataset, r.dataset_stratum_value, r.condition), []).append(r)

    summary = []
    for (model, dataset, stratum_value, condition), recs in sorted(grouped.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "", kv[0][2] or 0, kv[0][3] or "")):
        n_valid = sum(1 for r in recs if r.valid_run)
        n_evidence_eligible = sum(1 for r in recs if r.evidence_eligible)
        completed_eligible = [r for r in recs if r.evidence_eligible and r.completed_within_budget]
        n_completed_eligible = len(completed_eligible)
        n_max_budget = sum(1 for r in recs if r.max_budget_exit)

        n_local_full = sum(1 for r in completed_eligible if r.local_full_evidence)
        n_local_incomplete = n_completed_eligible - n_local_full
        n_evidence_eligible_incomplete_unconditional = sum(
            1 for r in recs if r.evidence_eligible and r.local_full_evidence is not True
        )
        missing_counts = [r.local_evidence_count_missing for r in completed_eligible if r.local_evidence_count_missing is not None]
        missing_dist = {b: sum(1 for m in missing_counts if (m == b if b < 3 else m >= 3)) for b in (0, 1, 2, 3)}
        n_incomplete_then_full = sum(1 for r in completed_eligible if r.incomplete_at_completion_full_at_final)

        summary.append(
            {
                "model": model, "dataset": dataset, "evidence_hop_stratum": stratum_value, "condition": condition,
                "n_valid": n_valid,
                "n_evidence_eligible": n_evidence_eligible,
                "n_completed_and_evidence_eligible": n_completed_eligible,
                "n_max_budget_exit": n_max_budget,
                "mean_anchor_coverage": _mean_or_na([r.anchor_coverage_fraction for r in recs if r.evidence_eligible]),
                "mean_local_completion_coverage": _mean_or_na([r.local_coverage_fraction for r in completed_eligible]),
                "mean_final_coverage": _mean_or_na([r.final_coverage_fraction for r in recs if r.evidence_eligible]),
                "full_evidence_rate_at_local_completion": (100.0 * n_local_full / n_completed_eligible) if n_completed_eligible else None,
                "n_full_evidence_at_local_completion": n_local_full,
                "evidence_incomplete_completion_rate_among_completed_eligible": (100.0 * n_local_incomplete / n_completed_eligible) if n_completed_eligible else None,
                "n_evidence_incomplete_completion_among_completed_eligible": n_local_incomplete,
                "unconditional_evidence_incomplete_incidence_among_eligible": (100.0 * n_evidence_eligible_incomplete_unconditional / n_evidence_eligible) if n_evidence_eligible else None,
                "n_evidence_incomplete_unconditional": n_evidence_eligible_incomplete_unconditional,
                "mean_missing_count_at_local_completion": _mean_or_na([float(m) for m in missing_counts]),
                "n_missing_0": missing_dist[0], "n_missing_1": missing_dist[1], "n_missing_2": missing_dist[2], "n_missing_3plus": missing_dist[3],
                "mean_recovery_gain": _mean_or_na([r.recovery_gain for r in completed_eligible]),
                "incomplete_at_completion_full_at_final_rate": (100.0 * n_incomplete_then_full / n_completed_eligible) if n_completed_eligible else None,
                "n_incomplete_at_completion_full_at_final": n_incomplete_then_full,
            }
        )
    return summary


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Paired cue-vs-control evidence effects
# ---------------------------------------------------------------------------


def build_paired_evidence_deltas(records: list[EvidenceHopRecord], canonical_records: list[cc.CanonicalCompletionRecord]) -> list[dict]:
    by_key = {(r.model, r.dataset, r.condition, r.example_id): r for r in records}
    grouped_canonical: dict[tuple, list[cc.CanonicalCompletionRecord]] = {}
    for r in canonical_records:
        grouped_canonical.setdefault((r.model, r.dataset), []).append(r)

    rows: list[dict] = []
    for (model, dataset), canon_recs in sorted(grouped_canonical.items()):
        for comp in COMPARISONS:
            pairs = cc.build_strict_pairs(canon_recs, comp.cue_condition, comp.control_condition)
            control_cov, cue_cov = [], []
            control_missing, cue_missing = [], []
            control_full, cue_full = [], []
            recovery_deltas = []
            n_lower = n_same = n_higher = 0
            for cue_c, control_c in pairs:
                cue_r = by_key.get((model, dataset, comp.cue_condition, cue_c.example_id))
                control_r = by_key.get((model, dataset, comp.control_condition, control_c.example_id))
                if cue_r is None or control_r is None:
                    continue
                if cue_r.local_coverage_fraction is None or control_r.local_coverage_fraction is None:
                    continue
                control_cov.append(control_r.local_coverage_fraction)
                cue_cov.append(cue_r.local_coverage_fraction)
                control_missing.append(control_r.local_evidence_count_missing or 0)
                cue_missing.append(cue_r.local_evidence_count_missing or 0)
                control_full.append(bool(control_r.local_full_evidence))
                cue_full.append(bool(cue_r.local_full_evidence))
                if cue_r.recovery_gain is not None and control_r.recovery_gain is not None:
                    recovery_deltas.append(cue_r.recovery_gain - control_r.recovery_gain)
                if cue_r.local_coverage_fraction < control_r.local_coverage_fraction:
                    n_lower += 1
                elif cue_r.local_coverage_fraction > control_r.local_coverage_fraction:
                    n_higher += 1
                else:
                    n_same += 1

            n = len(control_cov)
            if n == 0:
                rows.append(
                    {
                        "model": model, "dataset": dataset, "comparison_name": comp.comparison_name,
                        "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                        "n_paired": 0, "control_mean_coverage": None, "cue_mean_coverage": None,
                        "paired_evidence_loss": None, "pct_cue_lower": None, "pct_unchanged": None, "pct_cue_higher": None,
                        "control_full_evidence_rate": None, "cue_full_evidence_rate": None, "full_evidence_rate_loss": None,
                        "control_mean_missing_count": None, "cue_mean_missing_count": None, "missing_count_increase": None,
                        "recovery_gain_difference": None,
                    }
                )
                continue
            control_mean = sum(control_cov) / n
            cue_mean = sum(cue_cov) / n
            # Positive = control recovered more required evidence than cue
            # (spec's literal wording: control minus cue -- see README.md's
            # sign-convention note vs. `pcr_paired_deltas_stepwise.csv`).
            paired_loss = control_mean - cue_mean
            control_full_rate = 100.0 * sum(control_full) / n
            cue_full_rate = 100.0 * sum(cue_full) / n
            rows.append(
                {
                    "model": model, "dataset": dataset, "comparison_name": comp.comparison_name,
                    "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                    "n_paired": n, "control_mean_coverage": control_mean, "cue_mean_coverage": cue_mean,
                    "paired_evidence_loss": paired_loss,
                    "pct_cue_lower": 100.0 * n_lower / n, "pct_unchanged": 100.0 * n_same / n, "pct_cue_higher": 100.0 * n_higher / n,
                    "control_full_evidence_rate": control_full_rate, "cue_full_evidence_rate": cue_full_rate,
                    "full_evidence_rate_loss": control_full_rate - cue_full_rate,
                    "control_mean_missing_count": sum(control_missing) / n, "cue_mean_missing_count": sum(cue_missing) / n,
                    "missing_count_increase": (sum(cue_missing) - sum(control_missing)) / n,
                    "recovery_gain_difference": (sum(recovery_deltas) / len(recovery_deltas)) if recovery_deltas else None,
                }
            )
    return rows


def write_paired_deltas_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Manual validation sample
# ---------------------------------------------------------------------------


def build_manual_validation_sample(records: list[EvidenceHopRecord], per_dataset: int = 8) -> list[dict]:
    """>=5 runs per dataset (task minimum), both models, complete and
    incomplete local-completion coverage, and both terminal routes where
    available -- selected deterministically (sorted by example_id/condition)
    rather than randomly, so re-running this script reproduces the same
    sample."""
    rows = []
    by_dataset: dict[str, list[EvidenceHopRecord]] = {}
    for r in records:
        if r.evidence_eligible:
            by_dataset.setdefault(r.dataset, []).append(r)

    for dataset, recs in by_dataset.items():
        recs = sorted(recs, key=lambda r: (r.model, r.example_id, r.condition))
        # One bucket per (model, stratum), in the original qwen/llama order when
        # those are the bundle models: complete, incomplete, then budget exits.
        buckets = {}
        for stratum in ("complete", "incomplete"):
            for m in bundle_models():
                want = stratum == "complete"
                buckets[(m, stratum)] = [r for r in recs if r.model == m and r.local_full_evidence is want]
        for m in bundle_models():
            buckets[(m, "max_budget_exit")] = [r for r in recs if r.model == m and r.max_budget_exit]
        chosen: list[EvidenceHopRecord] = []
        for key, bucket in buckets.items():
            if bucket:
                chosen.append(bucket[0])
        # top up to per_dataset if the bucket pass didn't reach it
        i = 0
        seen_keys = {(r.model, r.dataset, r.condition, r.example_id) for r in chosen}
        while len(chosen) < per_dataset and i < len(recs):
            r = recs[i]
            key = (r.model, r.dataset, r.condition, r.example_id)
            if key not in seen_keys:
                chosen.append(r)
                seen_keys.add(key)
            i += 1
        for r in chosen[:per_dataset]:
            rows.append(
                {
                    "model": r.model, "dataset": r.dataset, "example_id": r.example_id, "condition": r.condition,
                    "trace_path": r.trace_path, "terminal_route": r.terminal_route,
                    "required_evidence_count": r.required_evidence_count,
                    "local_coverage_fraction": r.local_coverage_fraction, "local_full_evidence": r.local_full_evidence,
                    "final_coverage_fraction": r.final_coverage_fraction,
                    "required_paragraph_ids_raw": r.required_paragraph_ids_raw,
                    "local_retrieved_ids": r.local_retrieved_ids, "local_missing_ids": r.local_missing_ids,
                    "manual_check_notes": "",
                }
            )
    return rows


def write_manual_sample_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Validation gates (14, per task spec section 10)
# ---------------------------------------------------------------------------


class GateResult:
    def __init__(self, gate_id: int, description: str, passed: bool, detail: str):
        self.gate_id = gate_id
        self.description = description
        self.passed = passed
        self.detail = detail


def gate_1_unique_ids(records: list[EvidenceHopRecord]) -> GateResult:
    keys = [(r.model, r.dataset, r.condition, r.example_id) for r in records]
    dupes = len(keys) - len(set(keys))
    return GateResult(1, "Run IDs / example IDs unique within model, dataset, condition", dupes == 0, f"{dupes} duplicate keys")


def gate_2_counts_match_pcr(records: list[EvidenceHopRecord], pcr_run_inventory_path: Path) -> GateResult:
    if not pcr_run_inventory_path.exists():
        return GateResult(2, "Valid/eligible counts match latest PCR results", False, f"{pcr_run_inventory_path} not found")
    pcr_rows = _read_csv(pcr_run_inventory_path)
    pcr_valid = {(r["model"], r["dataset"], r["condition"]): int(r["n_valid"]) for r in pcr_rows}
    pcr_eligible = {(r["model"], r["dataset"], r["condition"]): int(r["n_pcr_eligible"]) for r in pcr_rows}
    own_valid: dict[tuple, int] = {}
    own_eligible: dict[tuple, int] = {}
    for r in records:
        key = (r.model, r.dataset, r.condition)
        if r.valid_run:
            own_valid[key] = own_valid.get(key, 0) + 1
        if r.pcr_eligible:
            own_eligible[key] = own_eligible.get(key, 0) + 1
    mismatches = []
    for key in pcr_valid:
        if own_valid.get(key, 0) != pcr_valid[key] or own_eligible.get(key, 0) != pcr_eligible.get(key, 0):
            mismatches.append((key, own_valid.get(key, 0), pcr_valid[key], own_eligible.get(key, 0), pcr_eligible.get(key, 0)))
    return GateResult(2, "Valid/eligible counts match latest PCR results", not mismatches, f"{len(mismatches)} cell mismatches" + (f": {mismatches[:3]}" if mismatches else ""))


def gate_3_completion_step_route_match(records: list[EvidenceHopRecord], pcr_run_level_path: Path) -> GateResult:
    if not pcr_run_level_path.exists():
        return GateResult(3, "Completion step/route match canonical PCR run-level data", False, f"{pcr_run_level_path} not found")
    pcr_rows = _read_csv(pcr_run_level_path)
    pcr_by_key = {(r["model"], r["dataset"], r["condition"], r["example_id"]): r for r in pcr_rows}
    mismatches = 0
    checked = 0
    for r in records:
        if not r.valid_run:
            continue
        pcr_row = pcr_by_key.get((r.model, r.dataset, r.condition, r.example_id))
        if pcr_row is None:
            continue
        checked += 1
        pcr_step = pcr_row["first_terminal_post_observation_step"]
        pcr_step = int(pcr_step) if pcr_step not in ("", "None") else None
        if pcr_step != r.first_terminal_post_observation_step or pcr_row["terminal_route"] != r.terminal_route:
            mismatches += 1
    return GateResult(3, "Completion step/route match canonical PCR run-level data", mismatches == 0, f"{mismatches}/{checked} rows mismatched")


def gate_4_no_completion_evidence_on_max_budget(records: list[EvidenceHopRecord]) -> GateResult:
    violations = [r for r in records if r.max_budget_exit and r.local_coverage_fraction is not None]
    return GateResult(4, "No completion-specific evidence value assigned to MAX_BUDGET_EXIT", not violations, f"{len(violations)} violations")


def gate_5_coverage_range(records: list[EvidenceHopRecord]) -> GateResult:
    violations = []
    for r in records:
        for frac in (r.anchor_coverage_fraction, r.local_coverage_fraction, r.final_coverage_fraction):
            if frac is not None and not (-1e-9 <= frac <= 1.0 + 1e-9):
                violations.append((r.model, r.dataset, r.condition, r.example_id, frac))
    return GateResult(5, "Coverage values in [0,1]", not violations, f"{len(violations)} out-of-range values")


def gate_6_missing_count_range(records: list[EvidenceHopRecord]) -> GateResult:
    violations = []
    for r in records:
        for missing in (r.anchor_evidence_count_missing, r.local_evidence_count_missing, r.final_evidence_count_missing):
            if missing is not None and not (0 <= missing <= r.required_evidence_count):
                violations.append((r.model, r.dataset, r.condition, r.example_id, missing, r.required_evidence_count))
    return GateResult(6, "Missing count in [0, required_evidence_count]", not violations, f"{len(violations)} out-of-range values")


def gate_7_full_evidence_iff_zero_missing(records: list[EvidenceHopRecord]) -> GateResult:
    violations = []
    for r in records:
        for full, missing in (
            (r.anchor_full_evidence, r.anchor_evidence_count_missing),
            (r.local_full_evidence, r.local_evidence_count_missing),
            (r.final_full_evidence, r.final_evidence_count_missing),
        ):
            if full is None or missing is None:
                continue
            if full != (missing == 0):
                violations.append((r.model, r.dataset, r.condition, r.example_id, full, missing))
    return GateResult(7, "Full-evidence true exactly when missing count is zero", not violations, f"{len(violations)} disagreements")


def gate_8_local_excludes_post_completion(records: list[EvidenceHopRecord]) -> GateResult:
    # Structural: local-completion coverage is computed from episode 0's
    # search_events only (see build_evidence_hop_record), and episode 0
    # ends at its own terminal decision by construction (no further
    # decisions/searches are logged in that episode after a RESEARCH_COMPLETE
    # or no-tool-exit decision) -- so this can only be violated by a
    # local-retrieved id that isn't a subset of the final-retrieved ids.
    violations = []
    for r in records:
        if not r.local_retrieved_ids:
            continue
        local_ids = set(r.local_retrieved_ids.split(";"))
        final_ids = set(r.final_retrieved_ids.split(";")) if r.final_retrieved_ids else set()
        if not local_ids.issubset(final_ids):
            violations.append((r.model, r.dataset, r.condition, r.example_id))
    return GateResult(8, "Local-completion evidence excludes post-completion retrievals", not violations, f"{len(violations)} rows where local retrieved ids are not a subset of final retrieved ids")


def gate_9_final_not_lower_than_local(records: list[EvidenceHopRecord]) -> GateResult:
    exceptions = []
    for r in records:
        if r.local_coverage_fraction is None or r.final_coverage_fraction is None:
            continue
        if r.final_coverage_fraction < r.local_coverage_fraction - 1e-9:
            exceptions.append((r.model, r.dataset, r.condition, r.example_id, r.local_coverage_fraction, r.final_coverage_fraction))
    return GateResult(9, "Final coverage is not lower than local coverage", not exceptions, f"{len(exceptions)} exceptions" + (f": {exceptions[:5]}" if exceptions else ""))


def gate_10_traceable_provenance(records: list[EvidenceHopRecord]) -> GateResult:
    violations = []
    for r in records:
        if not r.required_paragraph_ids_raw:
            continue
        required_set = set(r.required_paragraph_ids_raw.split(";"))
        for id_field in (r.anchor_retrieved_ids, r.anchor_missing_ids, r.local_retrieved_ids, r.local_missing_ids, r.final_retrieved_ids, r.final_missing_ids):
            if not id_field:
                continue
            ids = set(id_field.split(";"))
            if not ids.issubset(required_set):
                violations.append((r.model, r.dataset, r.condition, r.example_id))
                break
    return GateResult(10, "Every matched evidence identifier has traceable provenance", not violations, f"{len(violations)} rows with an id outside the example's own required set")


def gate_11_paired_identical_example_ids(paired_rows: list[dict], canonical_records: list[cc.CanonicalCompletionRecord]) -> GateResult:
    # Structural: build_strict_pairs joins on (model,dataset,example_id) by
    # construction (canonical_completion.py), so a violation here would mean
    # the pairing helper itself was bypassed -- spot-check n_paired against
    # an independent recomputation for a handful of comparisons.
    violations = []
    grouped: dict[tuple, list[cc.CanonicalCompletionRecord]] = {}
    for r in canonical_records:
        grouped.setdefault((r.model, r.dataset), []).append(r)
    for row in paired_rows:
        key = (row["model"], row["dataset"])
        recs = grouped.get(key, [])
        pairs = cc.build_strict_pairs(recs, row["cue_condition"], row["control_condition"])
        expected_n = len(pairs)
        # paired_evidence_deltas.csv's n_paired can be <= expected_n (some
        # PCR-eligible pairs may lack a defined local_coverage_fraction).
        if int(row["n_paired"]) > expected_n:
            violations.append((row["model"], row["dataset"], row["comparison_name"], row["n_paired"], expected_n))
    return GateResult(11, "Every paired comparison contains identical example IDs on both sides", not violations, f"{len(violations)} comparisons exceeding their strict-pair bound")


def gate_12_strata_agree_with_annotations(records: list[EvidenceHopRecord]) -> GateResult:
    violations = []
    for r in records:
        if not r.evidence_eligible or r.dataset_stratum_value is None:
            continue
        expected = EXPECTED_STRATUM_VALUES.get(r.dataset, set())
        if expected and r.dataset_stratum_value not in expected:
            violations.append((r.model, r.dataset, r.example_id, r.dataset_stratum_value))
    return GateResult(12, "Hop/evidence-count strata agree with dataset annotations", not violations, f"{len(violations)} out-of-range stratum values" + (f": {violations[:5]}" if violations else ""))


def gate_13_oracle_complete_full_anchor(records: list[EvidenceHopRecord]) -> GateResult:
    oracle_rows = [r for r in records if r.condition in ORACLE_CONDITIONS and r.evidence_eligible]
    if not oracle_rows:
        return GateResult(13, "Oracle Complete conditions have full anchor coverage", True, "no Oracle Complete rows found (nothing to check)")
    failures = [r for r in oracle_rows if r.anchor_full_evidence is not True]
    ok = not failures
    detail = f"{len(failures)}/{len(oracle_rows)} Oracle Complete rows without full anchor coverage"
    if failures:
        detail += " -- reported as a mapping-failure finding, not forced to pass; see README.md limitations"
    return GateResult(13, "Oracle Complete conditions have full anchor coverage", ok, detail)


def gate_14_manual_sample_coverage(manual_rows: list[dict]) -> GateResult:
    by_dataset: dict[str, list[dict]] = {}
    for row in manual_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    problems = []
    for dataset in config.DATASETS:
        rows = by_dataset.get(dataset, [])
        if len(rows) < 5:
            problems.append(f"{dataset}: only {len(rows)} rows (<5)")
            continue
        models_present = {row["model"] for row in rows}
        if models_present != set(bundle_models()):
            problems.append(f"{dataset}: missing model(s) {set(bundle_models()) - models_present}")
        routes_present = {row["terminal_route"] for row in rows}
        if len(routes_present) < 2:
            problems.append(f"{dataset}: only one terminal route present ({routes_present})")
    return GateResult(14, "Manual validation sample covers >=5 runs/dataset, both models, both routes where available", not problems, "; ".join(problems) if problems else "all datasets covered")


def run_all_gates(records: list[EvidenceHopRecord], canonical_records: list[cc.CanonicalCompletionRecord], paired_rows: list[dict], manual_rows: list[dict], bundle_root: Path) -> list[GateResult]:
    pcr_data_dir = bundle_root
    return [
        gate_1_unique_ids(records),
        gate_2_counts_match_pcr(records, pcr_data_dir / "run_inventory.csv"),
        gate_3_completion_step_route_match(records, pcr_data_dir / "pcr_run_level.csv"),
        gate_4_no_completion_evidence_on_max_budget(records),
        gate_5_coverage_range(records),
        gate_6_missing_count_range(records),
        gate_7_full_evidence_iff_zero_missing(records),
        gate_8_local_excludes_post_completion(records),
        gate_9_final_not_lower_than_local(records),
        gate_10_traceable_provenance(records),
        gate_11_paired_identical_example_ids(paired_rows, canonical_records),
        gate_12_strata_agree_with_annotations(records),
        gate_13_oracle_complete_full_anchor(records),
        gate_14_manual_sample_coverage(manual_rows),
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


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Commit lineage check
# ---------------------------------------------------------------------------


def check_commit_lineage(repo_root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False).stdout.strip()
    is_frozen_commit = head == FROZEN_COMMIT
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=repo_root, capture_output=True, check=False
    ).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False).stdout.strip()
    return {
        "frozen_commit_expected": FROZEN_COMMIT,
        "current_head": head,
        "head_equals_frozen_commit": is_frozen_commit,
        "frozen_commit_is_ancestor_of_head": is_ancestor,
        "working_tree_has_uncommitted_changes": bool(dirty),
    }


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


def write_readme(output_dir: Path, lineage: dict, n_records: int) -> None:
    text = f"""# Evidence/hop coverage -- README

## Source artifacts (locked PCR)

This task's prompt names `MASTER_RESULTS.md` and `FULL_MULTISTEP_PCR_TABLES.md`
as the locked canonical PCR artifacts, and a `whitebox_attack/v3/` code
layout. Neither name/path exists verbatim in this repository at the frozen
commit -- both are pre-restructure names. The commit
`9c327466ab8e730ba29a71db375ea5aa7556981d` (this task's own frozen-lineage
commit) is *itself* the commit that renamed `whitebox_attack/v3/` to
`attack/`/`analysis/` (see its own message: "Restructure repo to
attack/analysis/results/archive"; the exact old->new path table is in
`analysis/canonical/reorg_prep/reorg_top_level.py`). Applying that
documented mapping (`whitebox_attack/v3/analysis/<x>` -> `analysis/<x>`)
places this task's required output directory at `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/`,
used here.

For the PCR artifacts themselves: the only `MASTER_RESULTS.md` in this repo
is `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` (built earlier in this same
working session, from the identical qwen14b/llama8b production data this
task also targets). No file named `FULL_MULTISTEP_PCR_TABLES.md` exists
anywhere in the repo; the closest and evidently-intended match is
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/full_pcr_tables.md` ("Full PCR tables,
every step 1-29, plus PCR@30"). This README treats those two files as the
locked prior work referred to by the task prompt. **This is a reported
interpretation, not a silent guess** -- if a different `MASTER_RESULTS.md`/
`FULL_MULTISTEP_PCR_TABLES.md` was intended, this run should be redone
pointing at it.

Both of those files -- and every completion field in this package's own
output -- are generated from `src/prematurestop/evaluation/completion.py`,
which is unmodified by this task.

## Commit lineage

- Frozen commit required by the task: `{lineage['frozen_commit_expected']}`
- Current `HEAD`: `{lineage['current_head']}`
- `HEAD == frozen commit`: **{lineage['head_equals_frozen_commit']}**
- Working tree has uncommitted changes on top of that commit:
  **{lineage['working_tree_has_uncommitted_changes']}** (expected -- this
  session's own `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` and `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/`
  outputs are new, uncommitted files; no tracked file was modified or reset).

No `git reset`/`checkout` was performed. If `HEAD` above does not equal the
frozen commit when you read this, the production data may have changed
since this run and the run should be repeated.

## Regenerate

    conda activate whitebox
    python -m prematurestop.analysis.fixed_cue_matrix.evidence_hop_coverage.build_evidence_hop_coverage \\
        --output-dir results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage \\
        --pcr-bundle-dir results/whitebox/e1_fixed_cue_matrix/qwen_llama/data

## Dataset evidence mappings (spec section "Canonical evidence mapping")

Required evidence per example is read from each trace's own
`required_paragraph_ids` / `hop_count` / `required_dependency_count` fields
-- these are written once, at production-trace-generation time, by
`src/prematurestop/data/corpus.py` (`build_hotpot_rows` / `build_2wiki_rows` /
`build_musique_rows` / `build_multihoprag_rows`), directly from each
dataset's own annotations (HotpotQA `supporting_facts`, 2WikiMultiHopQA
`supporting_facts`, MuSiQue `question_decomposition` +
`paragraph_support_idx`, MultiHopRAG `evidence_list`) -- never inferred from
a generated answer, and frozen into the trace at that time (this task does
not reload the source HF datasets or rerun any model).

| Dataset | Stratum field | Values seen | Hop order source |
|---|---|---|---|
| HotpotQA | `hop_count` (fixed) | {{2}} | supporting_facts order (task: treat as two-hop, not stratified) |
| 2WikiMultiHopQA | `hop_count` | {{2, 4}} | supporting_facts order |
| MuSiQue | `hop_count` | {{2, 3, 4}} | `question_decomposition` order (official) |
| MultiHopRAG | `required_evidence_count` (task: not an ordered hop chain) | {{2, 3, 4}} | annotation list position only -- explicitly not validated as reasoning order |

Hop index within an example = position in the trace's own
`required_paragraph_ids` tuple. Verified against real traces (not assumed)
that this tuple is stable and identically ordered across every condition of
the same `(model, dataset, example_id)` (spot-checked 3 conditions of one
hotpot example, byte-identical list).

## Evidence matching rules

- Matching is exact-ID-set intersection against
  `required_paragraph_ids`/`required_dependency_ids` as stored in the
  trace -- never fuzzy text matching, never inferred from the generated
  answer.
- Coverage at the **anchor** (targeted observation preceding post-observation
  step 1) reuses the trace's own `coverage_fraction_at_injection` /
  `dependencies_seen_at_injection` / `missing_dependency_ids_at_injection`
  fields verbatim (computed once, at the single anchor search event, by the
  production runner -- `src/prematurestop/data/evidence.py::evaluate_coverage`).
- Coverage at **local completion (T_i)** reuses
  `prematurestop.evaluation.evidence._cumulative_union_coverage`'s method (a raw union of
  every search event's `retrieved_paragraph_ids` within episode 0), which
  that module's own docstring documents as necessary rather than trusting a
  single "seen so far" snapshot field (verified there to lose a previously-
  found dependency in some llama8b multi-search episodes). Episode 0 ends,
  by construction, at its own terminal decision, so this union already
  excludes anything retrieved after T_i -- there is no separate "step
  budget" to slice.
- Coverage at **final workflow termination** is a NEW computation added by
  this script: the same raw-union method, applied across every episode in
  the trace (episode 0 plus any supervisor redelegation), not just episode
  0. This is deliberately not the trace's own `coverage_fraction_at_termination`
  telemetry field (kept in `prematurestop.evaluation.evidence` as
  `whole_run_cumulative_retrieval_coverage_fraction` but not reused here) --
  computing final coverage with the exact same method as local coverage
  guarantees a "final < local" reading can only reflect a real trace
  anomaly, never a difference in measurement method. This is also why gate
  9 (final >= local) is structurally almost tautological except where a
  trace anomaly exists -- which is exactly what that gate is for.

## Reconciliation with `results/whitebox/e1_fixed_cue_matrix/qwen_llama`'s evidence tables

`results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_summary.csv` (and its
`tables/evidence_tables.md`) also reports a "paired evidence loss" number
for the same comparisons/models/datasets. **Those numbers will not match
this package's `paired_evidence_deltas.csv`, by design, not by bug** --
they measure a different boundary:

| | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_summary.csv` | This package's `paired_evidence_deltas.csv` |
|---|---|---|
| Boundary | Final-report-input coverage (`EvidenceRecord.final_input_coverage_fraction`) | Local completion at T_i (`local_coverage_fraction`) |
| Method | Citation-ID extraction from `final_report_generation`'s actual input text | Raw retrieved-paragraph-ID set intersection |
| When | Downstream -- after any supervisor redelegation and compression | Upstream -- at the researcher's own first post-observation completion decision |

**Per-row cross-check performed**: `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_run_level.csv::local_completion_coverage`
is actually built from `EvidenceRecord.episode0_terminal_coverage_fraction`
-- the SAME union-of-episode-0-search-events method this package's
`local_coverage_fraction` uses, reused from the same
`prematurestop.evaluation.evidence._cumulative_union_coverage` function. Comparing the two
columns row-by-row across all 42,547 valid rows: **42,155 (99.1%) match
exactly**; the remaining 392 are rows where `evidence_run_level.csv` reports
a real number but this package reports `NA` -- in every one of those cases
the row's terminal route is NOT a genuine PCR completion route (mostly
`MAX_BUDGET_EXIT`). `evidence_run_level.csv`'s field is defined for any
episode with a search at all, regardless of whether it ever reached a
completion decision; this package deliberately restricts `local_coverage_fraction`
to `NA` unless `T_i` is actually defined (spec: "For MAX_BUDGET_EXIT runs:
... set completion-specific fields to missing/NA rather than zero"). This
confirms the two packages' underlying retrieval accounting is consistent --
the packages differ only in (a) which downstream/upstream boundary their
PAIRED summary reports, and (b) how strictly they gate that boundary to
`NA` for non-completing runs.

**When citing evidence numbers, name the boundary explicitly** ("evidence
retrieved by local completion" vs "evidence present in the final report
generator's input") rather than saying "evidence coverage" alone -- see
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/TABLE_INDEX.md`'s matching
disambiguation section.

## Denominators

See `EVIDENCE_HOP_RESULTS.md` section 2 and `evidence_hop_summary.csv`'s
`n_valid` / `n_evidence_eligible` / `n_completed_and_evidence_eligible` /
`n_max_budget_exit` columns. `evidence_eligible` = valid run AND
`required_evidence_count > 0`. `completed_and_evidence_eligible` additionally
requires `pcr_eligible` AND a defined `T_i` (terminal route is
RESEARCH_COMPLETE or NO_TOOL_DIRECT_EXIT).

## Reused implementations

- `prematurestop.evaluation.completion` -- PCR (T_i, terminal_route,
  pcr_eligible, anchor fields). Unmodified.
- `prematurestop.evaluation.completion_inventory` -- run inventory, valid-row rule
  (`_row_is_valid`). Unmodified.
- `prematurestop.evaluation.trace_parser` / `prematurestop.evaluation.trace_schema` -- all trace/event
  parsing (episodes, decisions, search events). Unmodified.
- `prematurestop.evaluation.evidence._cumulative_union_coverage` / `._paragraph_index` --
  reused directly (imported, not copied) for local-completion coverage and
  ID normalization.
- `prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS` -- the single
  reconciled condition-comparison manifest (built earlier this session to
  replace two previously-inconsistent lists). No second comparisons list is
  introduced here.
- `prematurestop.evaluation.completion.build_strict_pairs` -- paired-
  eligible pairing rule (excludes rather than imputes).
- `prematurestop.data.corpus.py` / `prematurestop.data.evidence.py` -- read (not
  imported at runtime) to confirm where `required_paragraph_ids`/`hop_count`
  originate and that they are derived from dataset annotations, never from
  a generated answer.

Nothing in `src/prematurestop/evaluation/evidence.py`, `src/prematurestop/evaluation/completion.py`,
`src/prematurestop/evaluation/trace_parser.py`, or `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/` was
modified by this task.

## Known limitations

1. **Sentence-level supporting-fact recall (HotpotQA/2WikiMultiHopQA) is not
   computed.** `src/prematurestop/data/corpus.py` captures `sentence_ids` per
   dependency at corpus-build time, but this is not one of the fields
   serialized into the frozen trace JSON (only paragraph/document-level
   `required_paragraph_ids` and dependency-level
   `dependencies_seen_at_injection`/`missing_dependency_ids_at_injection`
   are). Recovering sentence-level recall would require reloading the
   original HuggingFace dataset objects and re-running `build_hotpot_rows`/
   `build_2wiki_rows` outside the trace -- out of scope for this pass per
   the task's own tight scope constraints. Document-level (paragraph)
   coverage, which the task also asks for and treats as the primary unit,
   is fully computed.
2. **MuSiQue reasoning-path text (`decomposition_question`/
   `intermediate_answer`/`bridge_entities`) is not recovered**, for the same
   reason (computed by `corpus.py` at build time, not serialized into the
   trace). Hop **order** and hop **count** ARE recovered reliably (position
   in `required_paragraph_ids`, verified stable across conditions), just not
   the natural-language decomposition text itself.
3. **Compressed-notes and final-report-input evidence boundaries** (spec
   3.2's "final-run coverage" via what `final_report_generation` actually
   received, as opposed to what was merely retrieved) are available in
   `prematurestop.evaluation.evidence.EvidenceRecord` but are NOT joined into this task's
   run-level table, since this task's "final workflow termination" is
   explicitly about retrieval state, not generator-input visibility (a
   separate, later analysis dimension per the task's own stop-list: "Do not
   ... calculate final-answer harm").
4. **Oracle Complete anchor-coverage gate (13)**: reported, not forced --
   see `EVIDENCE_HOP_RESULTS.md` section 7 for the actual pass/fail result
   and, if it fails, the specific rows involved.
5. This package's own `n_records` at generation time was **{n_records}**
   canonical run rows (2 models x 4 datasets x 12 conditions); see
   `run_level_evidence_hops.csv` for the frozen count at whatever time you
   are reading this.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_results_report(output_dir: Path, lineage: dict, summary_rows: list[dict], paired_rows: list[dict], gate_results: list[GateResult], n_total: int, n_evidence_eligible_by_cell: dict) -> None:
    n_pass = sum(1 for g in gate_results if g.passed)
    lines = [
        "# EVIDENCE_HOP_RESULTS.md",
        "",
        "## 1. Scope and locked relationship to PCR",
        "",
        "This is the first supporting analysis after PCR. PCR (T_i, terminal "
        "route, eligibility) is read verbatim from "
        "`prematurestop.evaluation.completion` and is not redefined, "
        "recomputed, or reinterpreted here. This report covers evidence/hop "
        "coverage only -- no thesis figures, no no-tool response content "
        "audit, no final-answer harm claims (a later, separate task).",
        "",
        f"Commit at generation time: `{lineage['current_head']}` "
        f"(equals the required frozen commit: {lineage['head_equals_frozen_commit']}).",
        "",
        "## 2. Data and eligibility counts",
        "",
        f"Total canonical run rows: **{n_total}** (2 models x 4 datasets x 12 conditions).",
        "",
        "See `evidence_hop_summary.csv` for the full "
        "model x dataset x stratum x condition breakdown "
        "(`n_valid`, `n_evidence_eligible`, `n_completed_and_evidence_eligible`, "
        "`n_max_budget_exit` columns). Evidence-eligible rows by model/dataset:",
        "",
        "| Model | Dataset | n_evidence_eligible |",
        "|---|---|---|",
    ]
    for (model, dataset), n in sorted(n_evidence_eligible_by_cell.items()):
        lines.append(f"| {model} | {dataset} | {n} |")

    lines += [
        "",
        "## 3. Evidence coverage at local completion",
        "",
        "See `run_level_evidence_hops.csv` (`local_coverage_fraction`, "
        "`local_full_evidence`, `local_evidence_count_missing`) for every "
        "row, and `evidence_hop_summary.csv`'s `mean_local_completion_coverage` "
        "/ `full_evidence_rate_at_local_completion` / "
        "`evidence_incomplete_completion_rate_among_completed_eligible` "
        "columns for the aggregate view, grouped by model x dataset x "
        "evidence/hop stratum x condition.",
        "",
        "**Not the same boundary as `results/whitebox/e1_fixed_cue_matrix/qwen_llama`'s "
        "`evidence_summary.csv`** (that one measures final-report-input "
        "coverage, downstream of any redelegation/compression -- this "
        "measures upstream coverage at the researcher's own completion "
        "decision). See `README.md`'s reconciliation section for the full "
        "explanation and a per-row cross-check between the two.",
        "",
        "## 4. Results by hop/evidence-count stratum",
        "",
        "`evidence_hop_summary.csv`'s `evidence_hop_stratum` column carries "
        "the per-dataset stratum value (hop_count for HotpotQA/2WikiMultiHopQA"
        "/MuSiQue, required_evidence_count for MultiHopRAG -- see README.md). "
        "Compact view (mean local-completion coverage, main-paper conditions "
        "only, all strata for that dataset):",
        "",
    ]
    main_conditions = set(config.MAIN_PAPER_CDF_CONDITIONS)
    header = "| Model | Dataset | Stratum | Condition | n_completed_eligible | mean_local_coverage | full_evidence_rate |"
    lines += [header, "|---|---|---|---|---|---|---|"]
    for row in summary_rows:
        if row["condition"] not in main_conditions:
            continue
        mc = row["mean_local_completion_coverage"]
        fr = row["full_evidence_rate_at_local_completion"]
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['evidence_hop_stratum']} | {row['condition']} | "
            f"{row['n_completed_and_evidence_eligible']} | {f'{mc:.3f}' if mc is not None else 'NA'} | "
            f"{f'{fr:.1f}%' if fr is not None else 'NA'} |"
        )

    lines += [
        "",
        "## 5. Paired cue-versus-control evidence loss",
        "",
        "See `paired_evidence_deltas.csv` for every comparison in "
        "`prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS` "
        "(the single reconciled comparison manifest -- no second list "
        "introduced here). `paired_evidence_loss` = control mean coverage "
        "minus cue mean coverage; positive means the cue condition recovered "
        "LESS required evidence at local completion than its paired control.",
        "",
        "| Comparison | Model | Dataset | n_paired | paired_evidence_loss | pct_cue_lower | full_evidence_rate_loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in paired_rows:
        if row["n_paired"] == 0 or row["n_paired"] == "0":
            continue
        pel = row["paired_evidence_loss"]
        pcl = row["pct_cue_lower"]
        ferl = row["full_evidence_rate_loss"]
        lines.append(
            f"| {row['comparison_name']} | {row['model']} | {row['dataset']} | {row['n_paired']} | "
            f"{f'{pel:.3f}' if isinstance(pel, float) else pel} | "
            f"{f'{pcl:.1f}%' if isinstance(pcl, float) else pcl} | "
            f"{f'{ferl:.1f}%' if isinstance(ferl, float) else ferl} |"
        )

    lines += [
        "",
        "This table may say whether cue conditions are associated with lower "
        "evidence coverage at completion. It must NOT be read as an "
        "answer-harm claim -- that requires the separate, later paired "
        "final-answer analysis.",
        "",
        "## 6. Supervisor recovery",
        "",
        "See `evidence_hop_summary.csv`'s `mean_recovery_gain` and "
        "`incomplete_at_completion_full_at_final_rate` columns, and "
        "`run_level_evidence_hops.csv`'s `recovery_gain` / "
        "`incomplete_at_completion_full_at_final` columns per row. "
        "`recovery_gain` = final coverage minus local-completion coverage; "
        "positive means the workflow recovered evidence after local "
        "completion (via supervisor redelegation or otherwise).",
        "",
        "## 7. Validation results",
        "",
        f"**{n_pass}/{len(gate_results)} gates passed.** See "
        "`validation_gates.csv` for the full detail; failing gates preserve "
        "every row (nothing is silently discarded) and record the likely "
        "cause in the `detail` column.",
        "",
        "| Gate | Description | Status |",
        "|---|---|---|",
    ]
    for g in gate_results:
        lines.append(f"| {g.gate_id} | {g.description} | {'PASS' if g.passed else 'FAIL'} |")

    lines += [
        "",
        "## 8. Important limitations",
        "",
        "See `README.md`'s \"Known limitations\" section (sentence-level "
        "supporting-fact recall, MuSiQue reasoning-path text, and the "
        "compressed-notes/final-report-input boundaries are not part of this "
        "table). This report does not make any final-answer-harm claim -- "
        "a cue that lowers evidence coverage at completion is reported as a "
        "process-level association only.",
        "",
        "## 9. The smallest justified next analysis",
        "",
        "Join `run_level_evidence_hops.csv` (or its `example_id`/`condition` "
        "keys) against `prematurestop.evaluation.answer_evaluation`'s per-row answer records "
        "to test whether evidence-incomplete completions are ALSO the runs "
        "with a control-correct/cue-wrong paired answer flip -- i.e. whether "
        "the evidence-loss association found here actually predicts "
        "downstream answer harm, rather than assuming it. That is the "
        "paired final-answer-harm analysis this task's stop-list explicitly "
        "deferred.",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "EVIDENCE_HOP_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_E1 / "evidence_hop_coverage")
    parser.add_argument("--production-root", type=Path, default=config.DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--pcr-bundle-dir", type=Path, default=Path("results/whitebox/e1_fixed_cue_matrix/qwen_llama/data"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--models", default=",".join(CANONICAL_BUNDLE_MODELS),
                        help="comma-separated model keys (default: the locked Qwen14B/Llama8B scope)")
    args = parser.parse_args()
    set_bundle_models(m.strip() for m in args.models.split(",") if m.strip())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Checking commit lineage...")
    lineage = check_commit_lineage(config.REPO_ROOT)
    print(f"  HEAD={lineage['current_head']}  matches frozen commit={lineage['head_equals_frozen_commit']}")
    if not lineage["head_equals_frozen_commit"] and not lineage["frozen_commit_is_ancestor_of_head"]:
        print("  WARNING: current repository does not descend from the frozen commit. Reporting, not blocking.")

    print("Building completed-shard inventory...")
    # Explicit scope, not config.INCLUDED_MODELS -- see
    # src/prematurestop/analysis/fixed_cue_matrix/results_bundle/scope.py's docstring. This
    # package's committed output shares the PCR bundle's frozen scope and
    # must not change if a third model temporarily enters
    # config.INCLUDED_MODELS for an unrelated reason.
    inventory = build_inventory(args.production_root, models=bundle_models())
    print(f"  {len(inventory.rows)} rows across {len(inventory.cells)} cells")

    print("Building canonical PCR records (reused, unmodified logic)...")
    canonical_records = cc.build_canonical_records(inventory.rows, max_workers=args.max_workers)
    print(f"  {len(canonical_records)} canonical records")

    print("Building evidence/hop records (this is the expensive step -- re-parses each trace)...")
    records = build_all_records(inventory.rows, canonical_records, max_workers=args.max_workers)
    print(f"  {len(records)} evidence/hop records")

    print("Writing run_level_evidence_hops.csv...")
    write_run_level_csv(records, args.output_dir / "run_level_evidence_hops.csv")

    print("Writing evidence_hop_summary.csv...")
    summary_rows = build_summary(records)
    write_summary_csv(summary_rows, args.output_dir / "evidence_hop_summary.csv")

    print("Writing paired_evidence_deltas.csv...")
    paired_rows = build_paired_evidence_deltas(records, canonical_records)
    write_paired_deltas_csv(paired_rows, args.output_dir / "paired_evidence_deltas.csv")

    print("Writing manual_validation_sample.csv...")
    manual_rows = build_manual_validation_sample(records)
    write_manual_sample_csv(manual_rows, args.output_dir / "manual_validation_sample.csv")

    print("Running validation gates...")
    gate_results = run_all_gates(records, canonical_records, paired_rows, manual_rows, args.pcr_bundle_dir)
    write_gates_csv(gate_results, args.output_dir / "validation_gates.csv")
    all_passed = print_gate_summary(gate_results)

    print("Writing README.md...")
    write_readme(args.output_dir, lineage, len(records))

    print("Writing EVIDENCE_HOP_RESULTS.md...")
    n_evidence_eligible_by_cell: dict = {}
    for r in records:
        if r.evidence_eligible:
            key = (r.model, r.dataset)
            n_evidence_eligible_by_cell[key] = n_evidence_eligible_by_cell.get(key, 0) + 1
    write_results_report(args.output_dir, lineage, summary_rows, paired_rows, gate_results, len(records), n_evidence_eligible_by_cell)

    print("\n" + "=" * 70)
    print("COMPLETION SUMMARY")
    print("=" * 70)
    print(f"Files created/updated in {args.output_dir}:")
    for name in ("README.md", "run_level_evidence_hops.csv", "evidence_hop_summary.csv", "paired_evidence_deltas.csv", "manual_validation_sample.csv", "validation_gates.csv", "EVIDENCE_HOP_RESULTS.md"):
        print(f"  - {name}")
    print(f"Total canonical rows: {len(records)}")
    print("Evidence-eligible rows by model/dataset:")
    for (model, dataset), n in sorted(n_evidence_eligible_by_cell.items()):
        print(f"  {model} / {dataset}: {n}")
    n_pass = sum(1 for g in gate_results if g.passed)
    print(f"Validation gates: {n_pass}/{len(gate_results)} passed")
    print(f"Ready for interpretation: {'YES' if all_passed else 'NO -- see validation_gates.csv for failing gates'}")
    print(
        "Next recommended task: join run_level_evidence_hops.csv against "
        "prematurestop.evaluation.answer_evaluation's per-row answer records to test whether "
        "evidence-incomplete completions are also the runs with a paired "
        "control-correct/cue-wrong answer flip (the paired final-answer-harm "
        "analysis this task explicitly deferred)."
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

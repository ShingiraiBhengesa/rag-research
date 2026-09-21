"""Step 7: `data/process_metrics.csv`.

Assembled directly from `CanonicalCompletionRecord` fields (already computed
by `pcr_full.py`'s step 3) joined with `EvidenceRecord` fields (`evidence_ext.py`'s
step 8) by (model, dataset, condition, example_id). The recovery-category
join/threshold logic is ported from
`analysis/final_paper_evaluation/recovery.py::build_supervisor_recovery_rows`
(imports only its label constants -- that function itself takes legacy
`PCRRecord`/`EvidenceRecord` types, not `CanonicalCompletionRecord`, so a
short fresh implementation against canonical types is lower-risk than
shimming legacy types through it).
"""

from __future__ import annotations

from pathlib import Path

from analysis.canonical import canonical_completion as cc
from analysis.evidence import EvidenceRecord
from analysis.final_paper_evaluation.recovery import (
    RECOVERY_NO_REDELEGATION,
    RECOVERY_REDELEGATED_FULL,
    RECOVERY_REDELEGATED_NO_RESTORE,
    RECOVERY_REDELEGATED_PARTIAL,
    RECOVERY_UNRESOLVED,
)


def _key(r) -> tuple[str, str, str, str]:
    return (r.model, r.dataset, r.condition, r.example_id)


def classify_recovery(
    supervisor_redelegated: bool, ep0_coverage: float | None, final_coverage: float | None
) -> str:
    """Verbatim port of recovery.py's threshold logic, against canonical
    field names (`supervisor_redelegated: bool` in place of
    `supervisor_redelegation_count: int > 0`)."""
    if not supervisor_redelegated:
        return RECOVERY_NO_REDELEGATION
    if ep0_coverage is None or final_coverage is None:
        return RECOVERY_UNRESOLVED
    if final_coverage >= 1.0 and ep0_coverage < 1.0:
        return RECOVERY_REDELEGATED_FULL
    if final_coverage > ep0_coverage + 1e-9:
        return RECOVERY_REDELEGATED_PARTIAL
    return RECOVERY_REDELEGATED_NO_RESTORE


def build_recovery_by_key(
    canonical_records: list[cc.CanonicalCompletionRecord], evidence_records: list[EvidenceRecord]
) -> dict[tuple[str, str, str, str], str]:
    ev_by_key = {_key(r): r for r in evidence_records}
    out: dict[tuple[str, str, str, str], str] = {}
    for r in canonical_records:
        ev = ev_by_key.get(_key(r))
        if ev is None:
            continue
        out[_key(r)] = classify_recovery(
            r.supervisor_redelegated, ev.episode0_terminal_coverage_fraction, ev.final_input_coverage_fraction
        )
    return out


def write_process_metrics_csv(
    canonical_records: list[cc.CanonicalCompletionRecord],
    recovery_by_key: dict[tuple[str, str, str, str], str],
    path: Path,
) -> None:
    rows = []
    for r in canonical_records:
        rows.append(
            {
                "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                "post_observation_researcher_decisions": r.post_observation_researcher_decisions,
                "post_observation_search_calls": r.post_observation_search_calls,
                "post_observation_think_calls": r.post_observation_think_calls,
                "local_completion_step": r.first_terminal_post_observation_step,
                "local_completion_route": r.terminal_route,
                "supervisor_redelegated": r.supervisor_redelegated,
                "later_evidence_recovery": recovery_by_key.get(_key(r), "unknown"),
                "final_system_status": r.final_status,
            }
        )
    cc._write_csv(path, rows)  # noqa: SLF001

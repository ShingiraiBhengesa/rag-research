"""Section 8/10: compression-loss and supervisor-recovery classification.

Joins the parent package's already-computed, corrected PCR records
(`pcr.PCRRecord`, for `supervisor_redelegation_count`) and evidence records
(`evidence.EvidenceRecord`, for the four coverage boundaries) by
(model, dataset, condition, example_id) -- both are independently validated
outputs, not re-derived here.

Compression loss: evidence retrieved during episode 0 (boundary 2) that is
not textually represented/traceable in the compressed notes handed to the
supervisor (boundary 3). Distinct from retrieval loss (episode 0 simply
didn't find it) and from unrecovered loss (still missing at boundary 4).

Supervisor recovery: only meaningful for rows where the supervisor actually
redelegated (`supervisor_redelegation_count > 0`); classified by whether
final-input coverage (boundary 4) actually improved on episode-0-terminal
coverage (boundary 2), not merely by whether redelegation happened.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis.evidence import EvidenceRecord
from analysis.pcr import PCRRecord

RECOVERY_NO_REDELEGATION = "no_redelegation_performed"
RECOVERY_REDELEGATED_NO_RESTORE = "redelegated_did_not_restore"
RECOVERY_REDELEGATED_PARTIAL = "redelegated_partially_restored"
RECOVERY_REDELEGATED_FULL = "redelegated_fully_restored"
RECOVERY_UNRESOLVED = "unresolved_missing_evidence_state"


def _key(model, dataset, condition, example_id):
    return (model, dataset, condition, example_id)


def build_compression_loss_rows(evidence_records: list[EvidenceRecord]) -> list[dict]:
    rows = []
    for r in evidence_records:
        if r.episode0_terminal_coverage_fraction is None:
            continue
        if r.compressed_notes_coverage_fraction is None:
            status = "NA_no_citations_in_compressed_notes"
            loss = None
        else:
            loss = r.episode0_terminal_coverage_fraction - r.compressed_notes_coverage_fraction
            status = "compression_loss" if loss > 1e-9 else ("no_compression_loss" if loss >= -1e-9 else "compressed_notes_exceeds_raw")
        rows.append(
            {
                "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                "episode0_terminal_coverage": r.episode0_terminal_coverage_fraction,
                "compressed_notes_coverage": r.compressed_notes_coverage_fraction,
                "compression_loss_fraction": loss,
                "status": status,
            }
        )
    return rows


def build_supervisor_recovery_rows(pcr_records: list[PCRRecord], evidence_records: list[EvidenceRecord]) -> list[dict]:
    ev_by_key = {_key(r.model, r.dataset, r.condition, r.example_id): r for r in evidence_records}
    rows = []
    for p in pcr_records:
        ev = ev_by_key.get(_key(p.model, p.dataset, p.condition, p.example_id))
        if ev is None:
            continue
        ep0 = ev.episode0_terminal_coverage_fraction
        final = ev.final_input_coverage_fraction
        redelegated = p.supervisor_redelegation_count > 0

        if not redelegated:
            category = RECOVERY_NO_REDELEGATION
        elif ep0 is None or final is None:
            category = RECOVERY_UNRESOLVED
        elif final >= 1.0 and ep0 < 1.0:
            category = RECOVERY_REDELEGATED_FULL
        elif final > ep0 + 1e-9:
            category = RECOVERY_REDELEGATED_PARTIAL
        else:
            category = RECOVERY_REDELEGATED_NO_RESTORE

        rows.append(
            {
                "model": p.model, "dataset": p.dataset, "condition": p.condition, "example_id": p.example_id,
                "supervisor_redelegation_count": p.supervisor_redelegation_count,
                "episode0_terminal_coverage": ep0, "final_input_coverage": final,
                "recovery_category": category,
            }
        )
    return rows


def summarize_recovery_by_cell(recovery_rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in recovery_rows:
        grouped[(r["model"], r["dataset"], r["condition"])].append(r)
    categories = (RECOVERY_NO_REDELEGATION, RECOVERY_REDELEGATED_NO_RESTORE, RECOVERY_REDELEGATED_PARTIAL, RECOVERY_REDELEGATED_FULL, RECOVERY_UNRESOLVED)
    summary = []
    for (model, dataset, condition), rs in grouped.items():
        n = len(rs)
        entry = {"model": model, "dataset": dataset, "condition": condition, "n": n}
        for cat in categories:
            cnt = sum(1 for r in rs if r["recovery_category"] == cat)
            entry[f"n_{cat}"] = cnt
            entry[f"pct_{cat}"] = 100.0 * cnt / n if n else None
        summary.append(entry)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

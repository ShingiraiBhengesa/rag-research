"""Step 2 (written using data from steps 3/8/9): `data/run_inventory.csv`.

One row per (model, dataset, condition) cell. `n_planned`/`n_completed` reuse
`completion_inventory.CellInventory` unchanged; `n_pcr_eligible`/
`n_evidence_eligible`/`n_answer_eligible` are computed here from the
already-built canonical/evidence/answer records (so this writer must be
called after those three record lists exist, even though it is
conceptually "step 2" in the spec's numbering).
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from prematurestop.evaluation.answer_evaluation import AnswerRecord
from prematurestop.evaluation.completion import CanonicalCompletionRecord
from prematurestop.evaluation.completion_inventory import CellInventory
from prematurestop.evaluation.evidence import EvidenceRecord


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_run_inventory_csv(
    cells: list[CellInventory],
    canonical_records: list[CanonicalCompletionRecord],
    evidence_records: list[EvidenceRecord],
    answer_records: list[AnswerRecord],
    path: Path,
    generated_at: str,
) -> None:
    n_pcr_eligible: dict[tuple[str, str, str], int] = {}
    for r in canonical_records:
        key = (r.model, r.dataset, r.condition)
        if r.pcr_eligible:
            n_pcr_eligible[key] = n_pcr_eligible.get(key, 0) + 1

    n_evidence_eligible: dict[tuple[str, str, str], int] = {}
    for r in evidence_records:
        key = (r.model, r.dataset, r.condition)
        if r.required_dependency_count > 0:
            n_evidence_eligible[key] = n_evidence_eligible.get(key, 0) + 1

    from prematurestop.evaluation.answer_evaluation import TECHNICAL_FAILURE_REASONS

    n_answer_eligible: dict[tuple[str, str, str], int] = {}
    for r in answer_records:
        key = (r.model, r.dataset, r.condition)
        if r.extraction_reason not in TECHNICAL_FAILURE_REASONS:
            n_answer_eligible[key] = n_answer_eligible.get(key, 0) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model", "dataset", "condition",
                "n_planned", "n_completed", "n_valid",
                "n_pcr_eligible", "n_evidence_eligible", "n_answer_eligible",
                "n_failures", "completion_fraction", "source_files", "generated_at",
            ]
        )
        for c in cells:
            key = (c.model, c.dataset, c.condition)
            writer.writerow(
                [
                    c.model, c.dataset, c.condition,
                    c.n_rows_expected, c.n_rows_present, c.n_rows_valid,
                    n_pcr_eligible.get(key, 0), n_evidence_eligible.get(key, 0), n_answer_eligible.get(key, 0),
                    c.n_rows_invalid,
                    (c.n_rows_present / c.n_rows_expected) if c.n_rows_expected else None,
                    f"runs/{c.model}/{c.dataset}/{c.condition}/*.jsonl",
                    generated_at,
                ]
            )

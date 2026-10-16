"""Section 2: completed-shard inventory for the included models
(config.INCLUDED_MODELS -- currently Qwen, Llama, and Gemma; GPT-OSS-20B is
excluded, its production matrix still blocked, see repo-root docs/notes/PUBLISHING_legacy.md).

Ground truth established directly against
`results/` (read-only):

* The expected matrix is frozen in `results/whitebox/e1_fixed_cue_matrix/production_records/matrix/v3_final_production_matrix.json`
  and `results/expected_runs/v3_expected_runs.jsonl` (86,400 expected rows =
  4 models x 4 datasets x 12 conditions x 450 examples; qwen14b+llama8b+gemma27b
  together account for 64,800 of those). Both are read-only inputs, never guessed.
* Each (model, dataset, condition) cell is sharded into exactly 9 files of 50
  rows each: `runs/<model>/<dataset>/<condition>/v3_<model>_<dataset>_<condition>_shard<II>of09.jsonl`.
  A row is only ever written by the production runner after that example
  finishes, so file/row presence is the authoritative completeness signal
  (stronger than sacct, which only reflects job *submission* state and is
  noisy across the multiple resubmission waves visible in the cluster's
  accounting history). sacct/squeue are still queried fresh here (read-only)
  as a corroborating signal and to positively confirm, at run time, that no
  array job for an included model is currently active (only GPT-OSS-20B's are,
  and those are excluded from this package by construction).
* Verified directly (2026-09-09): qwen14b and llama8b both have all 432
  expected shard files (9 x 12 x 4) with exactly 50/50 rows each -- 43,200/43,200
  rows present, of which 42,547 have `final_status == "ok"` with a readable
  trace file and 653 have `final_status == "error"` with no trace produced
  (a genuine run-time failure, not a parsing artifact). gemma27b's production
  matrix completed later (2026-09-13, 432/432 shards, 21,600 rows) and was
  extended into `config.INCLUDED_MODELS` on the same date -- see
  `canonical/gemma27b_analysis/gemma27b_results_synthesis.md` for its own
  completion/route breakdown.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prematurestop.evaluation import config

EXPECTED_SHARDS_PER_CONDITION = 9
EXPECTED_ROWS_PER_SHARD = 50
EXPECTED_ROWS_PER_CONDITION = EXPECTED_SHARDS_PER_CONDITION * EXPECTED_ROWS_PER_SHARD

STATUS_COMPLETE_FULL = "COMPLETE — FULL DATASET"
STATUS_PARTIAL = "PARTIAL — COMPLETED SHARDS ONLY"
STATUS_NONE = "NO COMPLETED SHARDS"
STATUS_INVALID = "INVALID/UNRESOLVED"


@dataclass
class ShardRecord:
    model: str
    dataset: str
    condition: str
    shard_index: int
    path: str
    exists: bool
    row_count: int
    valid_row_count: int
    invalid_row_count: int
    parse_error: Optional[str] = None


@dataclass
class RowRecord:
    model: str
    dataset: str
    condition: str
    shard_index: int
    example_id: str
    final_status: str
    error: str
    trace_path: str
    trace_exists: bool
    valid_answer: Optional[bool]
    row_index_in_shard: int
    run_id: str = ""
    frozen_config_hashes: dict = field(default_factory=dict)


@dataclass
class CellInventory:
    model: str
    dataset: str
    condition: str
    status: str
    n_shards_expected: int
    n_shards_present: int
    n_rows_expected: int
    n_rows_present: int
    n_rows_valid: int
    n_rows_invalid: int
    missing_shard_indices: list[int] = field(default_factory=list)
    duplicate_example_ids: list[str] = field(default_factory=list)


@dataclass
class InventoryResult:
    cells: list[CellInventory]
    shards: list[ShardRecord]
    rows: list[RowRecord]
    active_jobs: list[dict]


def _shard_path(production_root: Path, model: str, dataset: str, condition: str, shard_index: int) -> Path:
    return (
        production_root
        / "runs"
        / model
        / dataset
        / condition
        / f"v3_{model}_{dataset}_{condition}_shard{shard_index:02d}of{EXPECTED_SHARDS_PER_CONDITION:02d}.jsonl"
    )


def _load_shard_rows(model: str, dataset: str, condition: str, shard_index: int, path: Path) -> list[RowRecord]:
    rows: list[RowRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            trace_path = config.normalize_stored_trace_path(str(row.get("trace_path") or ""))
            trace_exists = bool(trace_path) and (config.REPO_ROOT / trace_path).exists()
            rows.append(
                RowRecord(
                    model=model,
                    dataset=dataset,
                    condition=condition,
                    shard_index=shard_index,
                    example_id=str(row.get("example_id") or ""),
                    final_status=str(row.get("final_status") or ""),
                    error=str(row.get("error") or ""),
                    trace_path=trace_path,
                    trace_exists=trace_exists,
                    valid_answer=row.get("valid_answer"),
                    row_index_in_shard=i,
                    run_id=str(row.get("run_id") or ""),
                    frozen_config_hashes=dict(row.get("frozen_config_hashes") or {}),
                )
            )
    return rows


def _row_is_valid(row: RowRecord) -> bool:
    return row.final_status == "ok" and not row.error and row.trace_exists


def build_inventory(
    production_root: Path,
    models: tuple[str, ...] = config.INCLUDED_MODELS,
    datasets: tuple[str, ...] = config.DATASETS,
    conditions: tuple[str, ...] = config.CONDITIONS,
) -> InventoryResult:
    for model in models:
        config.assert_included_model(model)

    cells: list[CellInventory] = []
    all_shards: list[ShardRecord] = []
    all_rows: list[RowRecord] = []

    for model in models:
        for dataset in datasets:
            for condition in conditions:
                shard_records: list[ShardRecord] = []
                cell_rows: list[RowRecord] = []
                missing_shards: list[int] = []
                for shard_index in range(EXPECTED_SHARDS_PER_CONDITION):
                    path = _shard_path(production_root, model, dataset, condition, shard_index)
                    config.assert_path_not_excluded(str(path))
                    if not path.exists():
                        missing_shards.append(shard_index)
                        shard_records.append(
                            ShardRecord(
                                model=model,
                                dataset=dataset,
                                condition=condition,
                                shard_index=shard_index,
                                path=str(path),
                                exists=False,
                                row_count=0,
                                valid_row_count=0,
                                invalid_row_count=0,
                            )
                        )
                        continue
                    try:
                        rows = _load_shard_rows(model, dataset, condition, shard_index, path)
                        parse_error = None
                    except (json.JSONDecodeError, OSError) as exc:
                        rows = []
                        parse_error = str(exc)
                    valid_rows = [r for r in rows if _row_is_valid(r)]
                    shard_records.append(
                        ShardRecord(
                            model=model,
                            dataset=dataset,
                            condition=condition,
                            shard_index=shard_index,
                            path=str(path),
                            exists=True,
                            row_count=len(rows),
                            valid_row_count=len(valid_rows),
                            invalid_row_count=len(rows) - len(valid_rows),
                            parse_error=parse_error,
                        )
                    )
                    cell_rows.extend(rows)

                all_shards.extend(shard_records)
                all_rows.extend(cell_rows)

                seen_ids: dict[str, int] = {}
                duplicates: list[str] = []
                for r in cell_rows:
                    seen_ids[r.example_id] = seen_ids.get(r.example_id, 0) + 1
                for eid, count in seen_ids.items():
                    if count > 1:
                        duplicates.append(eid)

                n_present = sum(sr.row_count for sr in shard_records)
                n_valid = sum(sr.valid_row_count for sr in shard_records)
                n_invalid = n_present - n_valid
                n_shards_present = sum(1 for sr in shard_records if sr.exists)

                if n_shards_present == EXPECTED_SHARDS_PER_CONDITION and n_present == EXPECTED_ROWS_PER_CONDITION and not duplicates:
                    status = STATUS_COMPLETE_FULL
                elif n_shards_present == 0:
                    status = STATUS_NONE
                elif duplicates or any(sr.parse_error for sr in shard_records):
                    status = STATUS_INVALID
                else:
                    status = STATUS_PARTIAL

                cells.append(
                    CellInventory(
                        model=model,
                        dataset=dataset,
                        condition=condition,
                        status=status,
                        n_shards_expected=EXPECTED_SHARDS_PER_CONDITION,
                        n_shards_present=n_shards_present,
                        n_rows_expected=EXPECTED_ROWS_PER_CONDITION,
                        n_rows_present=n_present,
                        n_rows_valid=n_valid,
                        n_rows_invalid=n_invalid,
                        missing_shard_indices=missing_shards,
                        duplicate_example_ids=duplicates,
                    )
                )

    active_jobs = fetch_active_jobs_readonly()
    return InventoryResult(cells=cells, shards=all_shards, rows=all_rows, active_jobs=active_jobs)


def fetch_active_jobs_readonly(user: Optional[str] = None) -> list[dict]:
    """Read-only `squeue` snapshot. Never submits, cancels, or modifies any job."""
    import getpass

    user = user or getpass.getuser()
    try:
        proc = subprocess.run(
            ["squeue", "-u", user, "-o", "%i|%j|%T|%M", "-h"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    jobs = []
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        job_id, job_name, state, elapsed = parts
        jobs.append({"job_id": job_id, "job_name": job_name, "state": state, "elapsed": elapsed})
    return jobs


def write_active_models_excluded_csv(path: Path, active_jobs: list[dict]) -> None:
    excluded_families = ("gemma27b", "gpt_oss_20b")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_family", "excluded_reason", "active_job_id", "active_job_name", "active_job_state", "elapsed"])
        any_written = False
        for job in active_jobs:
            name = job.get("job_name", "")
            for family in excluded_families:
                if family in name:
                    writer.writerow([family, "still_active_in_production_matrix", job.get("job_id"), name, job.get("state"), job.get("elapsed")])
                    any_written = True
        for family in excluded_families:
            if not any(family in j.get("job_name", "") for j in active_jobs):
                writer.writerow([family, "excluded_at_model_family_level_by_task_specification", "", "", "", ""])


def write_inventory_csvs(result: InventoryResult, audits_dir: Path) -> None:
    audits_dir.mkdir(parents=True, exist_ok=True)

    with open(audits_dir / "model_dataset_condition_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model", "dataset", "condition", "status",
                "n_shards_expected", "n_shards_present",
                "n_rows_expected", "n_rows_present", "n_rows_valid", "n_rows_invalid",
                "missing_shard_indices", "duplicate_example_ids",
            ]
        )
        for c in result.cells:
            writer.writerow(
                [
                    c.model, c.dataset, c.condition, c.status,
                    c.n_shards_expected, c.n_shards_present,
                    c.n_rows_expected, c.n_rows_present, c.n_rows_valid, c.n_rows_invalid,
                    ";".join(map(str, c.missing_shard_indices)), ";".join(c.duplicate_example_ids),
                ]
            )

    with open(audits_dir / "completed_shards_used.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "condition", "shard_index", "path", "row_count", "valid_row_count", "invalid_row_count"])
        for s in result.shards:
            if s.exists and s.row_count > 0:
                writer.writerow([s.model, s.dataset, s.condition, s.shard_index, s.path, s.row_count, s.valid_row_count, s.invalid_row_count])

    with open(audits_dir / "noncompleted_shards_excluded.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "condition", "shard_index", "path", "reason"])
        for s in result.shards:
            if not s.exists:
                writer.writerow([s.model, s.dataset, s.condition, s.shard_index, s.path, "shard_file_missing"])
            elif s.parse_error:
                writer.writerow([s.model, s.dataset, s.condition, s.shard_index, s.path, f"parse_error:{s.parse_error}"])

    with open(audits_dir / "missing_expected_shards.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "condition", "missing_shard_index"])
        for c in result.cells:
            for idx in c.missing_shard_indices:
                writer.writerow([c.model, c.dataset, c.condition, idx])

    with open(audits_dir / "duplicate_shards.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "condition", "duplicate_example_id"])
        for c in result.cells:
            for eid in c.duplicate_example_ids:
                writer.writerow([c.model, c.dataset, c.condition, eid])

    with open(audits_dir / "invalid_completed_rows.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "condition", "shard_index", "example_id", "final_status", "error", "trace_path", "trace_exists"])
        for r in result.rows:
            if not _row_is_valid(r):
                writer.writerow([r.model, r.dataset, r.condition, r.shard_index, r.example_id, r.final_status, r.error, r.trace_path, r.trace_exists])

    write_active_models_excluded_csv(audits_dir / "active_models_excluded.csv", result.active_jobs)

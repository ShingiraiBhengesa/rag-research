"""Section 8: strict branch-pairing audit.

Reuses the existing, active `attack.runtime.trace_schema.validate_paired_traces`
(imported read-only) as the authoritative pairing invariant -- it already
checks per-branch validity, duplicate example/condition traces, paired
replay-anchor consistency, and (critically) that every branch in the pair
shares the same `shared_prefix_hash` and `base_tool_message_hash` on its
first search event. This module adds: batch iteration over every
(model, dataset, example_id) group, a lightweight hash-only extraction path
(no full episode parsing needed for this check), and CSV reporting.

A pair is strictly comparable only if `validate_paired_traces` accepts it.
Any other unmatched/prefix-mismatched rows may still appear in
condition-level descriptive summaries (per Section 7/8), but must never be
labeled a paired effect.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis import config
from analysis.completion_inventory import RowRecord
from attack.runtime.trace_schema import TraceValidationError, validate_paired_traces

# The comparisons Section 7 actually needs (each attack condition against
# both of its baselines); NaturalClean/NaturalNeutral themselves don't need
# self-pairs.
BASELINE_CONDITIONS: tuple[str, ...] = ("NaturalClean", "NaturalNeutral")


@dataclass
class HashFingerprint:
    model: str
    dataset: str
    condition: str
    example_id: str
    trace_path: str
    shared_prefix_hash: Optional[str]
    base_tool_message_hash: Optional[str]
    suffix_hash: Optional[str]
    parse_error: Optional[str] = None


def _first_search_event(trace: dict) -> dict:
    for entry in trace.get("events", []):
        if entry.get("event_type") == "benchmark_search":
            return entry
    return {}


def extract_fingerprint(model: str, dataset: str, condition: str, example_id: str, trace_path: str) -> HashFingerprint:
    config.assert_path_not_excluded(trace_path)
    full_path = trace_path if Path(trace_path).is_absolute() else config.REPO_ROOT / trace_path
    try:
        with open(full_path, "r", encoding="utf-8") as fh:
            trace = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return HashFingerprint(model, dataset, condition, example_id, trace_path, None, None, None, parse_error=str(exc))
    se = _first_search_event(trace)
    return HashFingerprint(
        model=model,
        dataset=dataset,
        condition=condition,
        example_id=example_id,
        trace_path=trace_path,
        shared_prefix_hash=se.get("shared_prefix_hash"),
        base_tool_message_hash=se.get("base_tool_message_hash"),
        suffix_hash=se.get("suffix_hash"),
    )


def _load_raw_trace(trace_path: str) -> dict:
    full_path = trace_path if Path(trace_path).is_absolute() else config.REPO_ROOT / trace_path
    with open(full_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _process_one_example_group(args: tuple) -> dict:
    (model, dataset, example_id), by_cond = args
    return _process_example_group(model, dataset, example_id, by_cond)


def build_pairing_audit(rows: list[RowRecord], max_workers: int = 10) -> dict:
    """Groups valid rows by (model, dataset, example_id) and checks each
    attack/ContinueSearch condition against NaturalClean and NaturalNeutral.
    """
    by_example: dict[tuple[str, str, str], dict[str, RowRecord]] = {}
    for r in rows:
        by_example.setdefault((r.model, r.dataset, r.example_id), {})[r.condition] = r

    pairing_rows: list[dict] = []
    violation_rows: list[dict] = []
    hash_rows: list[dict] = []

    from concurrent.futures import ProcessPoolExecutor, as_completed

    items = list(by_example.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_one_example_group, item) for item in items]
        for fut in as_completed(futures):
            group_result = fut.result()
            pairing_rows.extend(group_result["pairing"])
            violation_rows.extend(group_result["violations"])
            hash_rows.extend(group_result["hashes"])

    return {"pairing": pairing_rows, "violations": violation_rows, "hashes": hash_rows}


def _process_example_group(model: str, dataset: str, example_id: str, by_cond: dict[str, RowRecord]) -> dict:
    pairing_rows: list[dict] = []
    violation_rows: list[dict] = []
    hash_rows: list[dict] = []
    if True:
        fingerprints: dict[str, HashFingerprint] = {}
        raw_traces: dict[str, dict] = {}
        load_errors: dict[str, str] = {}
        for cond, row in by_cond.items():
            config.assert_path_not_excluded(row.trace_path)
            try:
                raw_traces[cond] = _load_raw_trace(row.trace_path)
            except (OSError, json.JSONDecodeError) as exc:
                load_errors[cond] = str(exc)
                fingerprints[cond] = HashFingerprint(model, dataset, cond, example_id, row.trace_path, None, None, None, parse_error=str(exc))
                continue
            se = _first_search_event(raw_traces[cond])
            fp = HashFingerprint(
                model=model, dataset=dataset, condition=cond, example_id=example_id, trace_path=row.trace_path,
                shared_prefix_hash=se.get("shared_prefix_hash"), base_tool_message_hash=se.get("base_tool_message_hash"),
                suffix_hash=se.get("suffix_hash"),
            )
            fingerprints[cond] = fp
            hash_rows.append(
                {
                    "model": model, "dataset": dataset, "example_id": example_id, "condition": cond,
                    "shared_prefix_hash": fp.shared_prefix_hash, "base_tool_message_hash": fp.base_tool_message_hash,
                    "suffix_hash": fp.suffix_hash, "parse_error": fp.parse_error,
                }
            )

        for baseline in BASELINE_CONDITIONS:
            if baseline not in by_cond:
                continue
            for cond, row in by_cond.items():
                if cond == baseline:
                    continue
                target_fp = fingerprints.get(cond)
                base_fp = fingerprints.get(baseline)
                if target_fp is None or base_fp is None:
                    continue
                if target_fp.parse_error or base_fp.parse_error:
                    violation_rows.append(
                        {
                            "model": model, "dataset": dataset, "example_id": example_id,
                            "condition_a": baseline, "condition_b": cond,
                            "reason": f"trace_load_error: {target_fp.parse_error or base_fp.parse_error}",
                        }
                    )
                    continue

                hash_match = (
                    target_fp.shared_prefix_hash == base_fp.shared_prefix_hash
                    and target_fp.base_tool_message_hash == base_fp.base_tool_message_hash
                    and target_fp.shared_prefix_hash is not None
                )

                strict_ok = False
                reason = ""
                if hash_match:
                    try:
                        validate_paired_traces([raw_traces[baseline], raw_traces[cond]])
                        strict_ok = True
                    except TraceValidationError as exc:
                        reason = f"validate_paired_traces_failed: {exc}"
                else:
                    reason = "hash_mismatch_or_missing"

                pairing_rows.append(
                    {
                        "model": model, "dataset": dataset, "example_id": example_id,
                        "condition_a": baseline, "condition_b": cond,
                        "shared_prefix_hash_a": base_fp.shared_prefix_hash, "shared_prefix_hash_b": target_fp.shared_prefix_hash,
                        "base_tool_message_hash_a": base_fp.base_tool_message_hash, "base_tool_message_hash_b": target_fp.base_tool_message_hash,
                        "strictly_paired": strict_ok,
                    }
                )
                if not strict_ok:
                    violation_rows.append(
                        {
                            "model": model, "dataset": dataset, "example_id": example_id,
                            "condition_a": baseline, "condition_b": cond, "reason": reason,
                        }
                    )

    return {"pairing": pairing_rows, "violations": violation_rows, "hashes": hash_rows}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pairing_outputs(audit: dict, audits_dir: Path) -> None:
    _write_csv(audits_dir / "strict_pairing_audit.csv", audit["pairing"])
    _write_csv(audits_dir / "pairing_violations.csv", audit["violations"])
    _write_csv(audits_dir / "prefix_hash_audit.csv", audit["hashes"])

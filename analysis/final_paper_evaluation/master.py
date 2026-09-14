"""Section 3: one authoritative per-example master dataset.

One row per (model, dataset, condition, example_id), doing a single trace
parse per row and reusing the parent package's already-validated logic
directly (never re-implementing it): `trace_parser.parse_trace` for episode
structure, `evidence`'s boundary-computation helpers (the corrected union
and citation-normalization logic), and `answer_evaluator_v2.evaluate_answer_v2`
for scoring. A single pass per trace also guarantees every field on one row
is mutually consistent -- no join risk between separately-computed
datasets.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from analysis import config
from analysis.answer_evaluation import ManifestEntry, load_manifest
from analysis.completion_inventory import RowRecord
from analysis.evidence import (
    _cited_paragraph_ids,
    _cumulative_union_coverage,
    _extract_compress_research_text,
    _extract_final_report_generation_input_text,
    _normalized_required_paragraph_ids,
)
from analysis.trace_parser import load_trace_json, parse_trace
from attack.answer_scoring.answer_evaluator_v2 import EVALUATOR_VERSION, evaluate_answer_v2

TECHNICAL_FAILURE_EXTRACTION_REASONS = {"no_manifest_entry", "no_final_report"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MasterRow:
    model: str
    dataset: str
    condition: str
    example_id: str

    # Section 2 execution-validity / provenance
    shard_index: int
    run_id: str
    execution_valid: bool
    final_status: str
    error: str
    corpus_sha256: str
    evidence_sha256: str
    manifest_sha256: str

    # Identity / pairing hashes
    question_hash: Optional[str]
    decoding_config_hash: Optional[str]
    pre_injection_trace_hash: Optional[str]  # == shared_prefix_hash
    first_retrieved_observation_hash: Optional[str]  # == base_tool_message_hash
    branch_point_identifier: Optional[str]  # explicit alias of pre_injection_trace_hash

    # Injection / branch state
    cue_present: bool
    evidence_state_at_injection_hash: Optional[str]
    coverage_fraction_at_injection: Optional[float]
    missing_dependency_count_at_injection: Optional[int]

    # Research-control behavior (episode 0)
    branch_reached: bool
    no_branch_reason: Optional[str]
    first_post_injection_action: Optional[str]
    local_completion_route: Optional[str]
    local_completion_step: Optional[int]
    post_injection_search_count: int
    supervisor_redelegation_count: int

    # Evidence at every boundary (corrected methodology)
    episode0_terminal_coverage_fraction: Optional[float]
    compressed_notes_coverage_fraction: Optional[float]
    final_input_coverage_fraction: Optional[float]
    whole_run_cumulative_retrieval_coverage_fraction: Optional[float]

    # Final answer
    raw_final_report: str
    extracted_short_answer: Optional[str]
    extractable: bool
    extraction_reason: str
    semantic_label: str
    semantic_correct: bool
    exact_match: float
    token_f1: float
    gold_containment: bool
    gold_answer: str
    evaluator_version: str


def build_master_row(row: RowRecord, production_root: Path) -> MasterRow:
    config.assert_path_not_excluded(row.trace_path)
    full_path = Path(row.trace_path) if Path(row.trace_path).is_absolute() else config.REPO_ROOT / row.trace_path
    raw = load_trace_json(full_path)
    pt = parse_trace(raw, trace_path=str(full_path))
    ep0 = pt.first_episode

    manifest = load_manifest(row.dataset, production_root)
    entry: Optional[ManifestEntry] = manifest.get(row.example_id)
    question_hash = _sha256(entry.question) if entry else None

    model_config = raw.get("model")
    decoding_config_hash = (
        _sha256(json.dumps(model_config, sort_keys=True, default=str)) if isinstance(model_config, dict) else None
    )

    first_search = ep0.search_events[0] if (ep0 and ep0.search_events) else None
    shared_prefix_hash = first_search.raw.get("shared_prefix_hash") if first_search else None
    base_tool_message_hash = first_search.raw.get("base_tool_message_hash") if first_search else None
    cue_present = bool(first_search and first_search.attacked_observation)

    first_post_branch = ep0.post_branch_decisions[0] if (ep0 and ep0.post_branch_decisions) else None
    if first_post_branch is None:
        first_post_injection_action = None
    elif not first_post_branch.tool_call_names:
        first_post_injection_action = "no_tool_call"
    else:
        first_post_injection_action = ",".join(first_post_branch.tool_call_names)

    missing_at_injection = pt.missing_dependency_ids_at_injection
    evidence_state_hash = (
        _sha256(",".join(sorted(missing_at_injection))) if pt.coverage_fraction_at_injection is not None else None
    )

    no_branch_reason = None
    if ep0 is not None and not ep0.branch_reached:
        no_branch_reason = "search_tool_error" if ep0.search_attempted_but_failed else "no_search_attempted"

    ep0_had_search = bool(ep0 and ep0.search_events)
    if ep0_had_search:
        ep0_frac, _ = _cumulative_union_coverage(ep0)
    else:
        ep0_frac = 0.0 if pt.required_dependency_count else None

    required_paragraph_index_set = _normalized_required_paragraph_ids(pt.required_paragraph_ids)

    compressed_text = _extract_compress_research_text(raw.get("events", []))
    compressed_frac = None
    if compressed_text:
        cited = _cited_paragraph_ids(compressed_text)
        if cited and required_paragraph_index_set:
            compressed_frac = len(cited.intersection(required_paragraph_index_set)) / len(required_paragraph_index_set)

    final_input_text = _extract_final_report_generation_input_text(raw.get("events", []))
    final_input_frac = None
    if final_input_text:
        cited = _cited_paragraph_ids(final_input_text)
        if cited and required_paragraph_index_set:
            final_input_frac = len(cited.intersection(required_paragraph_index_set)) / len(required_paragraph_index_set)

    whole_run_frac = pt.coverage_fraction_at_termination

    if entry is not None and pt.reached_final_report_generation and pt.final_report:
        evaluation = evaluate_answer_v2(
            question=entry.question, raw_final_answer=pt.final_report,
            gold_answer=entry.answer, gold_aliases=entry.answer_aliases,
        )
        answer_fields = dict(
            extracted_short_answer=evaluation.extraction.short_answer,
            extractable=evaluation.extraction.extractable,
            extraction_reason=evaluation.extraction.reason,
            semantic_label=evaluation.semantic.semantic_label,
            semantic_correct=evaluation.semantic.semantic_correct,
            exact_match=evaluation.official_scores.exact_match,
            token_f1=evaluation.official_scores.token_f1,
            gold_containment=evaluation.gold_containment,
            gold_answer=entry.answer,
        )
    else:
        reason = "no_manifest_entry" if entry is None else "no_final_report"
        answer_fields = dict(
            extracted_short_answer=None, extractable=False, extraction_reason=reason,
            semantic_label="incorrect", semantic_correct=False, exact_match=0.0, token_f1=0.0,
            gold_containment=False, gold_answer=entry.answer if entry else "",
        )

    return MasterRow(
        model=row.model, dataset=row.dataset, condition=row.condition, example_id=row.example_id,
        shard_index=row.shard_index, run_id=row.run_id,
        execution_valid=(row.final_status == "ok" and not row.error and row.trace_exists),
        final_status=row.final_status, error=row.error,
        corpus_sha256=row.frozen_config_hashes.get("corpus_sha256", ""),
        evidence_sha256=row.frozen_config_hashes.get("evidence_sha256", ""),
        manifest_sha256=row.frozen_config_hashes.get("manifest_sha256", ""),
        question_hash=question_hash, decoding_config_hash=decoding_config_hash,
        pre_injection_trace_hash=shared_prefix_hash, first_retrieved_observation_hash=base_tool_message_hash,
        branch_point_identifier=shared_prefix_hash,
        cue_present=cue_present, evidence_state_at_injection_hash=evidence_state_hash,
        coverage_fraction_at_injection=pt.coverage_fraction_at_injection,
        missing_dependency_count_at_injection=len(missing_at_injection) if pt.coverage_fraction_at_injection is not None else None,
        branch_reached=bool(ep0 and ep0.branch_reached), no_branch_reason=no_branch_reason,
        first_post_injection_action=first_post_injection_action,
        local_completion_route=(ep0.terminal_route if ep0 else None),
        local_completion_step=(ep0.terminal_post_branch_index if ep0 else None),
        post_injection_search_count=(ep0.post_injection_search_count if ep0 else 0),
        supervisor_redelegation_count=pt.supervisor_redelegation_count,
        episode0_terminal_coverage_fraction=ep0_frac,
        compressed_notes_coverage_fraction=compressed_frac,
        final_input_coverage_fraction=final_input_frac,
        whole_run_cumulative_retrieval_coverage_fraction=whole_run_frac,
        raw_final_report=pt.final_report,
        evaluator_version=EVALUATOR_VERSION,
        **answer_fields,
    )


def _build_one_args(args: tuple) -> MasterRow:
    row, production_root_str = args
    return build_master_row(row, Path(production_root_str))


def build_master_rows(rows: list[RowRecord], production_root: Path, max_workers: int = 10) -> list[MasterRow]:
    args = [(row, str(production_root)) for row in rows]
    results: list[MasterRow] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_build_one_args, a) for a in args]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def write_master_csv(rows: list[MasterRow], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(MasterRow)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({name: getattr(r, name) for name in fieldnames})

"""Helpers for running one victim (Open Deep Research) branch and reading its trace.

Extracted unchanged from the archived adaptive-cue pilot: building a run request,
finding the first benchmark_search event in a trace, and re-deriving the clean
first observation (d1) for a set of first-search queries.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from prematurestop.system.runner import V3ModelConfig, V3RunRequest

CONTROL_CONDITION = "NaturalClean"


def _victim_request(
    *, dataset, example_id, question, condition, corpus_path, evidence_path, output_dir,
    model: V3ModelConfig, run_id, replay_anchor, cue_overrides, top_k, emergency_max_react_tool_calls,
    max_researcher_iterations,
) -> V3RunRequest:
    return V3RunRequest(
        dataset=dataset,
        example_id=example_id,
        question=question,
        condition=condition,
        corpus_path=Path(corpus_path),
        evidence_path=Path(evidence_path) if evidence_path else None,
        output_dir=Path(output_dir),
        model=model,
        top_k=top_k,
        retrieval_scope="per_example",
        cue_overrides=cue_overrides,
        emergency_max_react_tool_calls=emergency_max_react_tool_calls,
        max_researcher_iterations=max_researcher_iterations,
        allow_production_ids=True,
        run_id=run_id,
        replay_anchor=replay_anchor,
    )


def _first_search_event(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    for event in trace.get("events", []):
        if isinstance(event, Mapping) and event.get("event_type") == "benchmark_search":
            return event
    return {}


def _clean_observation_text(
    *, corpus_path, evidence_path, dataset, example_id, queries, output_dir, top_k
) -> str:
    """Deterministically re-derive the clean first observation the victim sees,
    for use as attacker input. Verified against the anchor run's hash by caller."""
    from prematurestop.system.benchmark_search import run_benchmark_search

    probe_cfg = {
        "configurable": {
            "benchmark_corpus_path": str(corpus_path),
            "benchmark_evidence_path": str(evidence_path) if evidence_path else "",
            "benchmark_dataset": dataset,
            "benchmark_example_id": example_id,
            "benchmark_retrieval_scope": "per_example",
            "benchmark_top_k": top_k,
            "benchmark_cache_dir": str(Path(output_dir) / "cache"),
            "v3_condition": CONTROL_CONDITION,
            "v3_run_id": f"pilot_clean_probe_{uuid.uuid4()}",
            "v3_cue_overrides": {},
        }
    }
    return run_benchmark_search(list(queries), config=probe_cfg)

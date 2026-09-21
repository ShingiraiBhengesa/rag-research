"""Guarded runner for ODR with benchmark-local retrieval.

This module defines infrastructure only. It does not submit cluster jobs and it
refuses production-looking example ids unless explicitly overridden later.
"""

from __future__ import annotations

import asyncio
import ast
import json
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from attack.runtime.cues import V3_CONDITIONS, suffix_for_condition
from attack.runtime.trace_schema import validate_paired_traces


@dataclass(frozen=True)
class V3ModelConfig:
    model_name: str
    base_url: str
    provider: str = "openai"
    api_key: str = "EMPTY"
    research_max_tokens: int = 1000
    compression_max_tokens: int = 2000
    final_report_max_tokens: int = 1000
    temperature: float = 0.0
    seed: int | None = 20260820
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class V3RunRequest:
    dataset: str
    example_id: str
    question: str
    condition: str
    corpus_path: Path
    evidence_path: Path | None
    output_dir: Path
    model: V3ModelConfig
    top_k: int = 5
    retrieval_scope: str = "per_example"
    cue_overrides: Mapping[str, str] | None = None
    emergency_max_react_tool_calls: int = 30
    max_researcher_iterations: int = 6
    allow_production_ids: bool = False
    run_id: str | None = None
    replay_anchor: Mapping[str, Any] | None = None


async def run_single_condition(request: V3RunRequest) -> dict[str, Any]:
    """Run one condition through the actual ODR LangGraph."""
    _guard_request(request)
    suffix_for_condition(request.condition, request.cue_overrides)

    from langgraph.checkpoint.memory import MemorySaver
    from open_deep_research.deep_researcher import deep_researcher_builder

    run_id = request.run_id or str(uuid.uuid4())
    branch_dir = request.output_dir / "branches" / request.dataset / request.example_id / request.condition
    branch_dir.mkdir(parents=True, exist_ok=True)
    search_trace_path = branch_dir / "benchmark_search_events.jsonl"
    trace_path = branch_dir / "trace.json"
    _reset_branch_artifacts(search_trace_path, trace_path)

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    runnable_config = build_runnable_config(request, run_id, search_trace_path)
    inputs = {"messages": [{"role": "user", "content": request.question}]}

    raw_events: list[dict[str, Any]] = []
    if hasattr(graph, "astream_events"):
        async for event in graph.astream_events(inputs, runnable_config, version="v2"):
            raw_events.append(_jsonable(event))
        final_state = _snapshot_values(graph, runnable_config)
    else:
        final_state = await graph.ainvoke(inputs, runnable_config)

    search_events = _read_jsonl(search_trace_path)
    final_report = _extract_final_report(final_state)
    extracted_answer = extract_answer_basic(final_report)
    coverage_summary = _coverage_summary_from_search_events(search_events)
    trace = {
        "schema_version": "v3_odr_benchmark_local_trace_v1",
        "run_id": run_id,
        "dataset": request.dataset,
        "example_id": request.example_id,
        "condition": request.condition,
        "model": asdict(request.model),
        "retrieval_scope": request.retrieval_scope,
        "top_k": request.top_k,
        "cue_overrides": dict(request.cue_overrides or {}),
        "paired_replay_anchor": _jsonable(request.replay_anchor) if request.replay_anchor else {},
        "events": _normalize_events(raw_events, search_events),
        "raw_odr_events": raw_events,
        **coverage_summary,
        "final_state": _jsonable(final_state),
        "final_report": final_report,
        "extracted_answer": extracted_answer,
        "trace_path": str(trace_path),
    }
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    return trace


async def run_paired_replay(requests: list[V3RunRequest]) -> list[dict[str, Any]]:
    """Run replay branches and reject the pair if shared-prefix hashes diverge."""
    if not requests:
        validate_paired_traces([])
        return []
    anchor_request = replace(
        requests[0],
        run_id=f"{requests[0].run_id or uuid.uuid4()}__paired_anchor_source",
        replay_anchor=None,
    )
    anchor_trace = await run_single_condition(anchor_request)
    anchor = _build_replay_anchor(anchor_trace)
    anchor_path = _write_replay_anchor(requests[0], anchor)
    anchor = {**anchor, "anchor_path": str(anchor_path)}

    traces = []
    for request in requests:
        trace = await run_single_condition(replace(request, replay_anchor=anchor))
        trace["paired_replay_anchor"] = anchor
        _persist_trace(trace)
        traces.append(trace)
    validate_paired_traces(traces)
    return traces


def build_runnable_config(request: V3RunRequest, run_id: str, search_trace_path: Path) -> dict[str, Any]:
    replay_anchor = dict(request.replay_anchor or {})
    recursion_limit = _recursion_limit_for_request(request)
    cfg: dict[str, Any] = {
        "thread_id": run_id,
        "allow_clarification": False,
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": request.max_researcher_iterations,
        "max_react_tool_calls": request.emergency_max_react_tool_calls,
        "search_api": "benchmark_local",
        "model_provider": request.model.provider,
        "model_base_url": request.model.base_url,
        "model_api_key": request.model.api_key,
        "model_temperature": request.model.temperature,
        "model_seed": request.model.seed,
        "model_reasoning_effort": request.model.reasoning_effort,
        "research_model": request.model.model_name,
        "research_model_max_tokens": request.model.research_max_tokens,
        "compression_model": request.model.model_name,
        "compression_model_max_tokens": request.model.compression_max_tokens,
        "final_report_model": request.model.model_name,
        "final_report_model_max_tokens": request.model.final_report_max_tokens,
        "summarization_model": request.model.model_name,
        "benchmark_corpus_path": str(request.corpus_path),
        "benchmark_evidence_path": str(request.evidence_path) if request.evidence_path else "",
        "benchmark_dataset": request.dataset,
        "benchmark_example_id": request.example_id,
        "benchmark_retrieval_scope": request.retrieval_scope,
        "benchmark_top_k": request.top_k,
        "benchmark_cache_dir": str(request.output_dir / "cache"),
        "benchmark_trace_path": str(search_trace_path),
        "v3_run_id": run_id,
        "v3_dataset": request.dataset,
        "v3_example_id": request.example_id,
        "v3_condition": request.condition,
        "v3_cue_overrides": dict(request.cue_overrides or {}),
        "v3_target_supervisor_delegation_index": 0,
        "v3_target_search_call_index": 0,
        "v3_replay_research_brief": replay_anchor.get("research_brief"),
        "v3_replay_research_topic": replay_anchor.get("research_topic"),
        "v3_replay_first_search_queries": list(replay_anchor.get("first_search_queries") or []),
        "v3_replay_supervisor_tool_call_id": replay_anchor.get("supervisor_tool_call_id"),
        "v3_replay_search_tool_call_id": replay_anchor.get("search_tool_call_id"),
    }
    return {"configurable": cfg, "recursion_limit": recursion_limit}


def _recursion_limit_for_request(request: V3RunRequest) -> int:
    """Keep LangGraph's recursion ceiling above the configured tool-call caps."""
    researcher_budget = int(request.emergency_max_react_tool_calls) * 4
    supervisor_budget = int(request.max_researcher_iterations) * 6
    return max(120, researcher_budget + supervisor_budget + 30)


def extract_answer_basic(final_report: str) -> str:
    for raw_line in final_report.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return ""


def _guard_request(request: V3RunRequest) -> None:
    if request.condition not in V3_CONDITIONS:
        raise ValueError(f"unsupported condition: {request.condition}")
    if not request.allow_production_ids and not request.example_id.startswith("dev_"):
        raise ValueError(
            f"refusing non-approved example id {request.example_id!r}; "
            "non-production validation must use dev_* ids"
        )
    if request.condition.startswith("OracleComplete") and request.dataset == "multihoprag" and not request.evidence_path:
        raise ValueError("MultiHopRAG oracle-complete conditions require a completed evidence audit")


def _normalize_events(raw_events: list[dict[str, Any]], search_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    step = 0
    for event in raw_events:
        normalized.append({"step": step, "event_type": "odr_event", "event": event})
        step += 1
    for event in search_events:
        normalized.append({"step": step, **event})
        step += 1
    return normalized


def _coverage_summary_from_search_events(search_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not search_events:
        return {
            "hop_count": 0,
            "required_dependency_count": 0,
            "required_paragraph_ids": [],
            "retrieved_paragraph_ids_at_termination": [],
            "covered_dependency_ids_at_termination": [],
            "dependencies_seen_at_injection": [],
            "dependencies_seen_at_termination": [],
            "missing_dependency_ids_at_injection": [],
            "missing_dependency_ids_at_termination": [],
            "coverage_fraction_at_injection": None,
            "coverage_fraction_at_termination": None,
            "first_step_full_coverage": None,
            "termination_before_full_coverage": None,
        }

    required_ids = list(search_events[0].get("required_dependency_ids") or [])
    required_paragraph_ids = list(search_events[0].get("required_paragraph_ids") or [])
    required_set = set(required_ids)
    cumulative_seen: set[str] = set()
    cumulative_paragraphs: set[str] = set()
    first_full_coverage = None
    injection_event = None

    for step, event in enumerate(search_events):
        cumulative_seen.update(
            str(dep_id)
            for dep_id in (
                event.get("covered_dependency_ids_after_observation")
                or event.get("dependencies_seen_after_observation")
                or ()
            )
        )
        cumulative_paragraphs.update(
            str(paragraph_id)
            for paragraph_id in event.get("retrieved_paragraph_ids_after_observation") or ()
        )
        if event.get("attacked_observation"):
            injection_event = event
        if required_set and first_full_coverage is None and required_set.issubset(cumulative_seen):
            first_full_coverage = step

    termination_seen = sorted(cumulative_seen, key=lambda dep_id: required_ids.index(dep_id) if dep_id in required_ids else len(required_ids))
    termination_missing = [dep_id for dep_id in required_ids if dep_id not in cumulative_seen]
    termination_fraction = len(set(termination_seen).intersection(required_set)) / len(required_ids) if required_ids else None
    termination_paragraphs = sorted(
        cumulative_paragraphs,
        key=lambda paragraph_id: (
            required_paragraph_ids.index(paragraph_id)
            if paragraph_id in required_paragraph_ids
            else len(required_paragraph_ids)
        ),
    )

    if injection_event is not None:
        injection_seen = list(injection_event.get("dependencies_seen_at_injection") or [])
        injection_missing = list(injection_event.get("missing_dependency_ids_at_injection") or [])
        injection_fraction = injection_event.get("coverage_fraction_at_injection")
    else:
        first_event = search_events[0]
        injection_seen = list(first_event.get("dependencies_seen_after_observation") or [])
        injection_missing = list(first_event.get("missing_dependency_ids_after_observation") or [])
        injection_fraction = first_event.get("coverage_fraction_after_observation")

    return {
        "hop_count": int(search_events[0].get("hop_count") or len(required_ids)),
        "required_dependency_count": int(search_events[0].get("required_dependency_count") or len(required_ids)),
        "required_paragraph_ids": required_paragraph_ids,
        "retrieved_paragraph_ids_at_termination": termination_paragraphs,
        "covered_dependency_ids_at_termination": termination_seen,
        "dependencies_seen_at_injection": injection_seen,
        "dependencies_seen_at_termination": termination_seen,
        "missing_dependency_ids_at_injection": injection_missing,
        "missing_dependency_ids_at_termination": termination_missing,
        "coverage_fraction_at_injection": injection_fraction,
        "coverage_fraction_at_termination": termination_fraction,
        "first_step_full_coverage": first_full_coverage,
        "termination_before_full_coverage": first_full_coverage is None,
    }


def _snapshot_values(graph: Any, config: Mapping[str, Any]) -> Any:
    try:
        snapshot = graph.get_state(config, subgraphs=True)
        return getattr(snapshot, "values", snapshot)
    except Exception:
        return {}


def _extract_final_report(final_state: Any) -> str:
    if isinstance(final_state, Mapping):
        return str(final_state.get("final_report") or "")
    values = getattr(final_state, "values", None)
    if isinstance(values, Mapping):
        return str(values.get("final_report") or "")
    return ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def _build_replay_anchor(trace: Mapping[str, Any]) -> dict[str, Any]:
    first_search = _first_search_event(trace)
    queries = [str(query) for query in first_search.get("queries") or []]
    research_brief = _extract_research_brief(trace)
    research_topic = _extract_first_conduct_research_topic(trace) or research_brief
    if not research_brief:
        raise RuntimeError("cannot build paired replay anchor without research_brief")
    if not queries:
        raise RuntimeError("cannot build paired replay anchor without first search queries")
    if not first_search.get("base_tool_message_hash") or not first_search.get("shared_prefix_hash"):
        raise RuntimeError("cannot build paired replay anchor without first search hashes")
    return {
        "schema_version": "v3_paired_replay_anchor_v1",
        "source_run_id": trace.get("run_id"),
        "source_condition": trace.get("condition"),
        "research_brief": research_brief,
        "research_topic": research_topic,
        "first_search_queries": queries,
        "source_search_query_hash": first_search.get("search_query_hash"),
        "source_base_tool_message_hash": first_search.get("base_tool_message_hash"),
        "source_shared_prefix_hash": first_search.get("shared_prefix_hash"),
        "supervisor_tool_call_id": first_search.get("supervisor_tool_call_id") or "v3_replay_conduct_research_0",
        "search_tool_call_id": first_search.get("search_tool_call_id") or "v3_replay_benchmark_search_0",
    }


def _write_replay_anchor(request: V3RunRequest, anchor: Mapping[str, Any]) -> Path:
    path = (
        request.output_dir
        / "paired_replay_anchors"
        / request.dataset
        / f"{request.example_id}_{anchor.get('source_shared_prefix_hash')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _persist_trace(trace: Mapping[str, Any]) -> None:
    trace_path = trace.get("trace_path")
    if trace_path:
        Path(str(trace_path)).write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")


def _first_search_event(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    for event in trace.get("events", []):
        if isinstance(event, Mapping) and event.get("event_type") == "benchmark_search":
            return event
    return {}


def _extract_research_brief(trace: Mapping[str, Any]) -> str:
    final_state = trace.get("final_state")
    if isinstance(final_state, Mapping):
        value = final_state.get("research_brief")
        if value:
            return str(value)
    value = trace.get("research_brief")
    return str(value) if value else ""


_RESEARCH_TOPIC_RE = re.compile(
    r"'name': 'ConductResearch'.*?'research_topic': (?P<literal>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")",
    re.DOTALL,
)


def _extract_first_conduct_research_topic(trace: Mapping[str, Any]) -> str:
    for event in trace.get("raw_odr_events", []):
        topic = _extract_topic_from_value(event)
        if topic:
            return topic
    final_state = trace.get("final_state")
    topic = _extract_topic_from_value(final_state)
    return topic or ""


def _extract_topic_from_value(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("name") == "ConductResearch":
            args = value.get("args")
            if isinstance(args, Mapping) and args.get("research_topic"):
                return str(args["research_topic"])
        for nested in value.values():
            topic = _extract_topic_from_value(nested)
            if topic:
                return topic
    elif isinstance(value, (list, tuple)):
        for nested in value:
            topic = _extract_topic_from_value(nested)
            if topic:
                return topic
    elif isinstance(value, str) and "ConductResearch" in value and "research_topic" in value:
        match = _RESEARCH_TOPIC_RE.search(value)
        if match:
            try:
                return str(ast.literal_eval(match.group("literal")))
            except (SyntaxError, ValueError):
                return ""
    return ""


def _reset_branch_artifacts(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return repr(value)


def run_nonproduction_pair_for_debug(requests: list[V3RunRequest]) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="v3_odr_debug_"):
        return asyncio.run(run_paired_replay(requests))

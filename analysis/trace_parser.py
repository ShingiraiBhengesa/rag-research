"""Parse a single production `trace.json` into a :class:`ParsedTrace`.

Ground truth for this parser (established empirically against real completed
qwen14b and llama8b production traces, not assumed; the gemma27b-specific
forced-search branch-point gap was found the same way -- see
``trace_schema.ResearcherDecision.triggered_forced_search`` and
``analysis/CHANGELOG.md``):

* ``trace["events"]`` is the canonical, already-validated event stream (the
  same list ``attack.runtime.trace_schema.validate_branch_trace`` checks
  for monotonic ``step`` numbers). It interleaves two kinds of records:
  ``event_type == "odr_event"`` (a raw LangGraph callback event, wrapped as
  ``{"step": int, "event_type": "odr_event", "event": {...}}``) and
  ``event_type == "benchmark_search"`` (a search-tool execution record with
  a dataset-native evidence-coverage snapshot already computed by the
  production runner). ``trace["raw_odr_events"]`` carries the same underlying LangGraph
  callbacks at higher granularity and was cross-checked to yield identical
  routing decisions; ``events`` is used here because it is the validated,
  smaller source.
* A `researcher` node's ``on_chain_start`` input carries
  ``v3_researcher_subgraph_id`` (e.g. ``"supervisor_iter_2:delegation_0:tool_call_..."``),
  which uniquely identifies one delegated-researcher (``ConductResearch``)
  episode and is stable across every decision inside that episode. Grouping
  by first-appearance order of this id gives episode order; episode 0 is
  the first attacked episode PCR is defined over.
* A `researcher` node's ``on_chain_end`` output is a `repr()` string of a
  LangGraph `Command(update={'researcher_messages': [AIMessage(...,
  tool_calls=[...])]}, goto=...)`. The produced `AIMessage.tool_calls` is
  the *authoritative* terminal-route signal: empty -> no-tool exit; contains
  a call named ``"ResearchComplete"`` -> explicit ResearchComplete. This
  matches `open_deep_research/deep_researcher.py`'s own `researcher_tools`
  routing logic (`has_tool_calls` / `research_complete_called` checks),
  which decides purely from this same tool-call list. Verified against both
  Qwen (hermes parser) and Llama (llama3_json parser) traces.
* The branch point ("shared first-search observation") is defined by the
  first `benchmark_search` event *actually executed* inside an episode
  (grounded in real tool execution), not by assuming a fixed decision index
  -- a researcher may call `think_tool` one or more times before its first
  search.
* CAUTION (verified empirically, not documented anywhere upstream):
  ``trace["events"]`` is built by the runner's ``_normalize_events`` as
  ``[*all odr_events with step 0..N-1, *all benchmark_search entries with
  step N..N+M-1]`` -- i.e. every `benchmark_search` entry's own ``step``
  field is *not* its true chronological position, only its position in a
  counter appended after every odr_event. The relative *order* of
  `benchmark_search` entries among themselves is still chronological. To
  recover each search's true step, this parser zips the ordered list of
  `benchmark_search` entries against the ordered list of `on_tool_end`
  odr_events for the `tavily_search` tool (same count, same relative order
  in every sample checked) and uses the tool_end's real step instead.
* ``trace["model"]`` is the *model client config dict* (api_key, base_url,
  model_name, ...), not the short model alias ("qwen14b"/"llama8b"). The
  alias is recovered from the trace path
  (``.../traces/<model_alias>/branches/...``), the one place it appears as
  a plain string, and cross-checked against ``trace["model"]["model_name"]``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from analysis import config
from analysis.repr_parsing import LiteralScanError, extract_all_kwarg_occurrences, extract_last_kwarg
from analysis.trace_schema import (
    ParsedTrace,
    ResearcherDecision,
    ResearcherEpisode,
    SearchEvent,
)
from attack.runtime.production_models import FINAL_MODEL_IDS

RESEARCH_COMPLETE_TOOL_NAME = config.RESEARCH_COMPLETE_TOOL_NAME
SEARCH_TOOL_NAME = config.SEARCH_TOOL_NAME

_MODEL_NAME_TO_ALIAS = {v: k for k, v in FINAL_MODEL_IDS.items()}


def infer_model_alias_from_path(trace_path: str) -> str | None:
    parts = Path(trace_path).parts
    if "traces" in parts:
        idx = parts.index("traces")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


class TraceParseError(ValueError):
    """Raised for structural failures that make a trace unusable (not per-decision parse noise)."""


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(v) for v in value)


def load_trace_json(path: Path) -> dict:
    config.assert_path_not_excluded(str(path))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_odr_events(events: Iterable[Mapping[str, Any]]) -> Iterable[tuple[int, dict]]:
    for entry in events:
        if entry.get("event_type") == "odr_event":
            ev = entry.get("event") or {}
            step = entry.get("step")
            if step is None:
                step = ev.get("step")
            yield int(step), ev


def _parse_decision(step: int, subgraph_id: str | None, chain_end_event: Mapping[str, Any]) -> ResearcherDecision:
    output = chain_end_event.get("data", {}).get("output")
    text = _as_text(output)
    try:
        extracted = extract_last_kwarg(text, "tool_calls")
    except LiteralScanError as exc:
        return ResearcherDecision(
            step=step,
            subgraph_id=subgraph_id,
            tool_call_names=[],
            tool_calls_raw=None,
            has_tool_calls=None,
            is_research_complete=False,
            is_no_tool_exit=False,
            parse_ok=False,
            parse_error=str(exc),
        )
    if extracted is None:
        return ResearcherDecision(
            step=step,
            subgraph_id=subgraph_id,
            tool_call_names=[],
            tool_calls_raw=None,
            has_tool_calls=None,
            is_research_complete=False,
            is_no_tool_exit=False,
            parse_ok=False,
            parse_error="no tool_calls= kwarg found in researcher chain_end output",
        )
    tool_calls = extracted.value
    if not isinstance(tool_calls, list):
        return ResearcherDecision(
            step=step,
            subgraph_id=subgraph_id,
            tool_call_names=[],
            tool_calls_raw=None,
            has_tool_calls=None,
            is_research_complete=False,
            is_no_tool_exit=False,
            parse_ok=False,
            parse_error=f"tool_calls literal was not a list: {type(tool_calls)!r}",
        )
    names = [str(tc.get("name")) for tc in tool_calls if isinstance(tc, dict)]
    has_tool_calls = len(tool_calls) > 0
    return ResearcherDecision(
        step=step,
        subgraph_id=subgraph_id,
        tool_call_names=names,
        tool_calls_raw=tool_calls,
        has_tool_calls=has_tool_calls,
        is_research_complete=RESEARCH_COMPLETE_TOOL_NAME in names,
        is_no_tool_exit=not has_tool_calls,
        parse_ok=True,
    )


def _parse_search_event(real_step: int, entry: Mapping[str, Any]) -> SearchEvent:
    return SearchEvent(
        step=real_step,
        subgraph_id=entry.get("researcher_subgraph_id"),
        supervisor_delegation_index=entry.get("supervisor_delegation_index"),
        attacked_observation=bool(entry.get("attacked_observation")),
        coverage_fraction_after_observation=entry.get("coverage_fraction_after_observation"),
        dependencies_seen_after_observation=_tuple_of_str(entry.get("dependencies_seen_after_observation")),
        missing_dependency_ids_after_observation=_tuple_of_str(entry.get("missing_dependency_ids_after_observation")),
        required_dependency_ids=_tuple_of_str(entry.get("required_dependency_ids")),
        required_paragraph_ids=_tuple_of_str(entry.get("required_paragraph_ids")),
        retrieved_paragraph_ids=_tuple_of_str(entry.get("retrieved_paragraph_ids_after_observation")),
        normalized_queries=_tuple_of_str(entry.get("normalized_queries")),
        raw=dict(entry),
    )


def _find_forced_search_steps(events: list[Mapping[str, Any]]) -> list[int]:
    """Steps of `researcher_tools` on_chain_end events whose output contains
    a synthetic `forced_benchmark_local_search_<n>` tool-call id -- i.e. ODR's
    force_benchmark_search safety net actually executing a search, as
    distinct from the researcher model issuing one itself. See
    deep_researcher.py:_forced_benchmark_local_search_messages (id format
    `f"forced_benchmark_local_search_{iteration}"`, fired from two call
    sites: before any tool execution when the model's response had no tool
    calls at all, and after tool execution when none of the model's tool
    calls was a search). Uses `extract_all_kwarg_occurrences` (not just the
    last) because the second call site can append the forced AIMessage
    after the model's own (non-search) tool outputs in the same event.
    """
    steps: list[int] = []
    for step, ev in _iter_odr_events(events):
        if ev.get("name") != "researcher_tools" or ev.get("event") != "on_chain_end":
            continue
        output = ev.get("data", {}).get("output")
        text = _as_text(output)
        try:
            occurrences = extract_all_kwarg_occurrences(text, "tool_calls")
        except LiteralScanError:
            continue
        found = False
        for occ in occurrences:
            if not isinstance(occ.value, list):
                continue
            for tc in occ.value:
                if isinstance(tc, dict) and str(tc.get("id", "")).startswith("forced_benchmark_local_search"):
                    found = True
                    break
            if found:
                break
        if found:
            steps.append(step)
    return steps


def build_episodes(events: list[Mapping[str, Any]]) -> list[ResearcherEpisode]:
    # 1. Collect researcher on_chain_start (for subgraph id) and on_chain_end
    #    (for the decision payload), matched by position within trace order.
    researcher_starts: list[tuple[int, dict]] = []
    researcher_ends: list[tuple[int, dict]] = []
    for step, ev in _iter_odr_events(events):
        if ev.get("name") != "researcher":
            continue
        kind = ev.get("event")
        if kind == "on_chain_start":
            researcher_starts.append((step, ev))
        elif kind == "on_chain_end":
            researcher_ends.append((step, ev))

    if len(researcher_starts) != len(researcher_ends):
        raise TraceParseError(
            f"researcher on_chain_start/on_chain_end count mismatch: "
            f"{len(researcher_starts)} starts vs {len(researcher_ends)} ends"
        )

    decisions: list[ResearcherDecision] = []
    for (start_step, start_ev), (end_step, end_ev) in zip(researcher_starts, researcher_ends):
        subgraph_id = start_ev.get("data", {}).get("input", {}).get("v3_researcher_subgraph_id")
        decisions.append(_parse_decision(end_step, subgraph_id, end_ev))

    # 1b. Attribute ODR's force_benchmark_search safety net (see
    #     deep_researcher.py:_forced_benchmark_local_search_messages) to the
    #     decision whose empty tool-call list triggered it. The synthetic
    #     forced search is logged on the FOLLOWING `researcher_tools`
    #     on_chain_end event, never on a `researcher` on_chain_end event, so
    #     it would otherwise be invisible to branch-point detection --
    #     discovered auditing gemma27b, whose model almost always emits a
    #     "```tool_code" text block instead of a real structured tool call,
    #     so this safety net (rather than a genuine model tool call) is what
    #     actually produces its first search in nearly every episode.
    #     Attribution rule: each forced-search step is matched to the last
    #     decision whose step precedes it, relying on the same
    #     "episodes execute strictly sequentially" invariant already used
    #     for search_tool_error attribution below.
    forced_search_steps = _find_forced_search_steps(events)
    if forced_search_steps:
        for forced_step in forced_search_steps:
            owner_idx = None
            for i, d in enumerate(decisions):
                if d.step < forced_step:
                    owner_idx = i
                else:
                    break
            if owner_idx is not None:
                decisions[owner_idx] = replace(decisions[owner_idx], triggered_forced_search=True)

    # 2. Collect benchmark_search events. Their own "step" field is NOT their
    #    true chronological position (see module docstring) -- recover the
    #    real step by zipping against the ordered tavily_search on_tool_end
    #    odr_events, which share the same count and relative order.
    search_tool_end_steps = [
        step
        for step, ev in _iter_odr_events(events)
        if ev.get("name") == SEARCH_TOOL_NAME and ev.get("event") == "on_tool_end"
    ]
    bench_entries = [entry for entry in events if entry.get("event_type") == "benchmark_search"]
    if len(search_tool_end_steps) != len(bench_entries):
        raise TraceParseError(
            f"tavily_search on_tool_end count ({len(search_tool_end_steps)}) does not match "
            f"benchmark_search entry count ({len(bench_entries)}); cannot recover true search step order"
        )
    search_events: list[SearchEvent] = [
        _parse_search_event(real_step, entry) for real_step, entry in zip(search_tool_end_steps, bench_entries)
    ]

    # 3. Group into episodes by subgraph_id, preserving first-appearance order.
    episode_order: list[str | None] = []
    for decision in decisions:
        if decision.subgraph_id not in episode_order:
            episode_order.append(decision.subgraph_id)

    episodes: list[ResearcherEpisode] = []
    for order_idx, subgraph_id in enumerate(episode_order):
        ep_decisions = [d for d in decisions if d.subgraph_id == subgraph_id]
        ep_search_events = [s for s in search_events if s.subgraph_id == subgraph_id]
        supervisor_iter = None
        if isinstance(subgraph_id, str) and subgraph_id.startswith("supervisor_iter_"):
            try:
                supervisor_iter = int(subgraph_id.split(":", 1)[0].removeprefix("supervisor_iter_"))
            except ValueError:
                supervisor_iter = None
        episodes.append(
            ResearcherEpisode(
                subgraph_id=subgraph_id,
                episode_order=order_idx,
                supervisor_iter=supervisor_iter,
                decisions=ep_decisions,
                search_events=ep_search_events,
            )
        )

    # 4. Attribute search-tool `on_tool_error` events (e.g. Llama's
    #    stringified-list `queries` arg failing Pydantic validation) to the
    #    episode they occurred in. Episodes execute strictly sequentially
    #    (a redelegation only starts after the prior episode fully reaches
    #    compress_research), so an error at step S belongs to the first
    #    episode (in order) whose own decisions include a step >= S.
    error_steps = [
        step for step, ev in _iter_odr_events(events) if ev.get("name") == SEARCH_TOOL_NAME and ev.get("event") == "on_tool_error"
    ]
    if error_steps and episodes:
        episode_max_steps = [max((d.step for d in ep.decisions), default=-1) for ep in episodes]
        for error_step in error_steps:
            for ep, ep_max_step in zip(episodes, episode_max_steps):
                if ep_max_step >= error_step:
                    ep.search_tool_error_count += 1
                    break
            else:
                episodes[-1].search_tool_error_count += 1

    return episodes


def parse_trace(trace: Mapping[str, Any], *, trace_path: str) -> ParsedTrace:
    events = trace.get("events")
    if not isinstance(events, list):
        raise TraceParseError("trace['events'] is missing or not a list")

    parse_errors: list[str] = []
    try:
        episodes = build_episodes(events)
    except TraceParseError as exc:
        parse_errors.append(str(exc))
        episodes = []

    final_report = trace.get("final_report") or ""
    reached_final_report_generation = bool(final_report)

    model_alias = infer_model_alias_from_path(trace_path)
    model_config = trace.get("model")
    if isinstance(model_config, dict):
        declared_model_name = model_config.get("model_name")
        declared_alias = _MODEL_NAME_TO_ALIAS.get(declared_model_name)
        if declared_alias is not None and model_alias is not None and declared_alias != model_alias:
            parse_errors.append(
                f"model alias mismatch: path implies {model_alias!r}, "
                f"trace['model']['model_name']={declared_model_name!r} implies {declared_alias!r}"
            )
        elif model_alias is None:
            model_alias = declared_alias
    elif isinstance(model_config, str) and model_config:
        # Tolerate a future schema where trace["model"] is already the alias string.
        if model_alias is not None and model_config != model_alias:
            parse_errors.append(
                f"model alias mismatch: path implies {model_alias!r}, trace['model']={model_config!r}"
            )
        else:
            model_alias = model_alias or model_config

    return ParsedTrace(
        model=str(model_alias or ""),
        dataset=str(trace.get("dataset") or ""),
        condition=str(trace.get("condition") or ""),
        example_id=str(trace.get("example_id") or ""),
        trace_path=trace_path,
        schema_version=trace.get("schema_version"),
        episodes=episodes,
        reached_final_report_generation=reached_final_report_generation,
        final_report=final_report,
        extracted_answer=trace.get("extracted_answer") or "",
        coverage_fraction_at_injection=trace.get("coverage_fraction_at_injection"),
        coverage_fraction_at_termination=trace.get("coverage_fraction_at_termination"),
        dependencies_seen_at_injection=_tuple_of_str(trace.get("dependencies_seen_at_injection")),
        dependencies_seen_at_termination=_tuple_of_str(trace.get("dependencies_seen_at_termination")),
        missing_dependency_ids_at_injection=_tuple_of_str(trace.get("missing_dependency_ids_at_injection")),
        missing_dependency_ids_at_termination=_tuple_of_str(trace.get("missing_dependency_ids_at_termination")),
        required_dependency_count=int(trace.get("required_dependency_count") or 0),
        required_paragraph_ids=_tuple_of_str(trace.get("required_paragraph_ids")),
        retrieved_paragraph_ids_at_termination=_tuple_of_str(trace.get("retrieved_paragraph_ids_at_termination")),
        hop_count=int(trace.get("hop_count") or 0),
        parse_errors=parse_errors,
    )


def parse_trace_file(path: Path) -> ParsedTrace:
    trace = load_trace_json(path)
    return parse_trace(trace, trace_path=str(path))


def tool_call_query_strings(tool_call: Mapping[str, Any]) -> list[str]:
    """Best-effort extraction of search query text, handling Llama's stringified list args."""
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return []
    queries = args.get("queries")
    if isinstance(queries, list):
        return [str(q) for q in queries]
    if isinstance(queries, str):
        import ast as _ast

        try:
            parsed = _ast.literal_eval(queries)
        except Exception:  # noqa: BLE001
            return [queries]
        if isinstance(parsed, list):
            return [str(q) for q in parsed]
        return [queries]
    return []

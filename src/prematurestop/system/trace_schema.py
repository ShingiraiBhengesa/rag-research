"""Trace validation helpers for paired ODR runs."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from prematurestop.cues.fixed.registry import suffix_for_condition


class TraceValidationError(ValueError):
    """Raised when a trace violates a predeclared invariant."""


def validate_branch_trace(trace: Mapping[str, object]) -> None:
    condition = str(trace.get("condition") or "")
    cue = suffix_for_condition(condition, trace.get("cue_overrides") if isinstance(trace.get("cue_overrides"), Mapping) else None)

    events = trace.get("events")
    if not isinstance(events, list):
        raise TraceValidationError("trace events must be a list")

    _validate_monotonic_steps(events)

    injections = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("attacked_observation")
    ]
    if condition == "NaturalClean":
        if injections:
            raise TraceValidationError("NaturalClean must not contain an attacked observation")
    elif cue.requires_injection and len(injections) != 1:
        raise TraceValidationError(f"{condition} must contain exactly one primary injection")

    if injections:
        observed_suffix_hash = injections[0].get("suffix_hash")
        if observed_suffix_hash != cue.suffix_hash:
            raise TraceValidationError(
                f"suffix hash mismatch for {condition}: {observed_suffix_hash} != {cue.suffix_hash}"
            )

    if trace.get("final_report_required", True) and not trace.get("final_report"):
        raise TraceValidationError("trace is missing final_report")

    if trace.get("extracted_answer_required", True) and "extracted_answer" not in trace:
        raise TraceValidationError("trace is missing extracted_answer")

    if _has_benchmark_search_event(trace) and trace.get("final_report") and not _has_forwarded_research_notes(trace):
        raise TraceValidationError("trace reached final_report without forwarded research notes")


def validate_paired_traces(traces: Iterable[Mapping[str, object]]) -> None:
    traces = list(traces)
    if not traces:
        raise TraceValidationError("no paired traces supplied")
    example_keys = Counter((trace.get("dataset"), trace.get("example_id"), trace.get("condition")) for trace in traces)
    duplicates = [key for key, count in example_keys.items() if count > 1]
    if duplicates:
        raise TraceValidationError(f"duplicate example-condition traces: {duplicates}")

    for trace in traces:
        validate_branch_trace(trace)

    _validate_paired_replay_anchors(traces)

    prefix_hashes = {
        _first_search_event(trace).get("shared_prefix_hash")
        for trace in traces
        if _first_search_event(trace)
    }
    if len(prefix_hashes) > 1:
        raise TraceValidationError(f"paired branches have divergent prefix hashes: {sorted(prefix_hashes)}")

    base_hashes = {
        _first_search_event(trace).get("base_tool_message_hash")
        for trace in traces
        if _first_search_event(trace)
    }
    if len(base_hashes) > 1:
        raise TraceValidationError(f"paired branches have divergent base ToolMessage hashes: {sorted(base_hashes)}")


def _validate_monotonic_steps(events: list[object]) -> None:
    step_numbers = []
    for event in events:
        if isinstance(event, Mapping) and "step" in event:
            step_numbers.append(int(event["step"]))
    if step_numbers != sorted(step_numbers):
        raise TraceValidationError(f"non-monotonic step ordering: {step_numbers}")


def _first_search_event(trace: Mapping[str, object]) -> Mapping[str, object]:
    for event in trace.get("events", []):
        if isinstance(event, Mapping) and event.get("event_type") == "benchmark_search":
            return event
    return {}


def _has_benchmark_search_event(trace: Mapping[str, object]) -> bool:
    return bool(_first_search_event(trace))


def _has_forwarded_research_notes(trace: Mapping[str, object]) -> bool:
    final_state = trace.get("final_state")
    if isinstance(final_state, Mapping) and _state_has_forwarded_research_notes(final_state):
        return True
    return _final_report_input_had_forwarded_research_notes(trace)


def _state_has_forwarded_research_notes(state: Mapping[str, object]) -> bool:
    raw_notes = state.get("raw_notes")
    if _nonempty_note_list(raw_notes):
        return True
    notes = state.get("notes")
    return _nonempty_note_list(notes)


def _nonempty_note_list(value: object) -> bool:
    return isinstance(value, list) and any(str(note).strip() for note in value)


def _final_report_input_had_forwarded_research_notes(trace: Mapping[str, object]) -> bool:
    raw_events = trace.get("raw_odr_events")
    if not isinstance(raw_events, list):
        return False
    for event in raw_events:
        if not isinstance(event, Mapping):
            continue
        if _raw_event_node_name(event) != "final_report_generation":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        node_input = data.get("input")
        if isinstance(node_input, Mapping) and _state_has_forwarded_research_notes(node_input):
            return True
    return False


def _raw_event_node_name(event: Mapping[str, object]) -> str:
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        node_name = str(metadata.get("langgraph_node") or "")
        if node_name:
            return node_name
    return str(event.get("name") or "")


def _validate_paired_replay_anchors(traces: list[Mapping[str, object]]) -> None:
    anchors = [trace.get("paired_replay_anchor") for trace in traces]
    if not any(anchors):
        return
    if not all(isinstance(anchor, Mapping) for anchor in anchors):
        raise TraceValidationError("paired replay anchor must be present on every branch")

    anchor_keys = {
        (
            tuple(anchor.get("first_search_queries") or ()),
            anchor.get("source_base_tool_message_hash"),
            anchor.get("source_shared_prefix_hash"),
        )
        for anchor in anchors
        if isinstance(anchor, Mapping)
    }
    if len(anchor_keys) != 1:
        raise TraceValidationError("paired branches reference divergent replay anchors")

    for trace, anchor in zip(traces, anchors):
        if not isinstance(anchor, Mapping):
            raise TraceValidationError("paired replay anchor must be a mapping")
        event = _first_search_event(trace)
        if not event:
            raise TraceValidationError("paired replay branch has no benchmark search event")
        if list(event.get("queries") or []) != list(anchor.get("first_search_queries") or []):
            raise TraceValidationError("paired replay branch did not reuse anchor first-search queries")
        if event.get("base_tool_message_hash") != anchor.get("source_base_tool_message_hash"):
            raise TraceValidationError("paired replay branch did not reuse anchor base ToolMessage")
        if event.get("shared_prefix_hash") != anchor.get("source_shared_prefix_hash"):
            raise TraceValidationError("paired replay branch did not reproduce anchor shared-prefix hash")

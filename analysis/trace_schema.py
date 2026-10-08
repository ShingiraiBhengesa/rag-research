"""Typed records for parsed production traces.

This module only defines data shapes. The actual event-walking logic lives
in :mod:`analysis.trace_parser`. Kept separate so the
schema (what a "researcher decision", "episode", or "parsed trace" *is*) can
be read and reasoned about independently of how it is extracted from the
raw LangGraph event stream.

Provenance: reuses the invariant-checking helpers in the existing, active
``attack.runtime.trace_schema`` module (``validate_branch_trace`` /
``validate_paired_traces``) rather than re-implementing them -- that module
is imported read-only and is never modified by this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Re-exported so callers can validate raw traces with the same invariants
# the production runner itself enforces, without importing runtime code.
from attack.runtime.trace_schema import (  # noqa: F401
    TraceValidationError,
    validate_branch_trace,
    validate_paired_traces,
)


@dataclass
class ResearcherDecision:
    """One `researcher` node invocation (one researcher-model decision)."""

    step: int
    subgraph_id: Optional[str]
    tool_call_names: list[str]
    tool_calls_raw: Optional[list[dict]]
    has_tool_calls: Optional[bool]  # None only if parsing failed
    is_research_complete: bool
    is_no_tool_exit: bool
    parse_ok: bool
    parse_error: Optional[str] = None
    # True iff this decision's own (empty) tool call list caused ODR's
    # `force_benchmark_search` safety net to intercept and execute a search
    # on this decision's behalf (deep_researcher.py:_forced_benchmark_local_search_messages,
    # synthetic tool-call id `forced_benchmark_local_search_<n>`, logged on
    # the *following* `researcher_tools` on_chain_end event, never on this
    # decision's own `researcher` on_chain_end event). Discovered 2026-09-13
    # auditing gemma27b, whose model almost never emits a real structured
    # tool call (it writes a "```tool_code" text block instead) -- without
    # this flag, `_branch_decision_index` below could never find a branch
    # point for such an episode at all. See trace_parser.py's
    # `_find_forced_search_steps` for how this is detected.
    triggered_forced_search: bool = False
    # True iff tool_calls is genuinely empty (has_tool_calls/is_no_tool_exit
    # above are NOT changed by this flag -- they reflect what actually,
    # literally happened) AND the model's own content text imitates ODR's
    # gemma-history-rendering format ("Assistant requested tool call:\n{...}",
    # deep_researcher.py::_plain_turn_from_message -- shown to the model as
    # *prior*-turn history, then imitated on a *later* turn) instead of a
    # real tool call. Discovered 2026-09-28 auditing gemma27b: 93% of its
    # sampled NO_TOOL_DIRECT_EXIT decisions were this pattern -- but the
    # imitated call is confirmed (by trace inspection) to never actually
    # execute: the episode goes straight to compress_research with no
    # further researcher decision, exactly as if it really had no tool
    # call. This is a technical parser/prompt-interaction confound, not
    # evidence either way of the researcher's true intent-to-stop -- see
    # canonical_completion.classify_local_terminal_route, which routes a
    # terminal decision carrying this flag to TOOL_OR_PARSER_FAILURE
    # (excluded from PCR) rather than counting it as a real NO_TOOL_DIRECT_EXIT.
    reconstructed_from_imitated_history_text: bool = False


@dataclass
class SearchEvent:
    """One `benchmark_search` tool execution, with its coverage snapshot."""

    step: int
    subgraph_id: Optional[str]
    supervisor_delegation_index: Optional[int]
    attacked_observation: bool
    coverage_fraction_after_observation: Optional[float]
    dependencies_seen_after_observation: tuple[str, ...]
    missing_dependency_ids_after_observation: tuple[str, ...]
    required_dependency_ids: tuple[str, ...]
    required_paragraph_ids: tuple[str, ...]
    retrieved_paragraph_ids: tuple[str, ...]
    normalized_queries: tuple[str, ...]
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class ResearcherEpisode:
    """One delegated researcher subgraph invocation (one `ConductResearch` call)."""

    subgraph_id: Optional[str]
    episode_order: int  # 0 = first (attacked) episode; 1.. = later redelegations
    supervisor_iter: Optional[int]
    decisions: list[ResearcherDecision]
    search_events: list[SearchEvent]
    # Count of `on_tool_error` events for the search tool attributed to this
    # episode (e.g. Llama emitting a stringified-list `queries` arg that
    # fails the tool's Pydantic schema). A genuine, systematic technical
    # failure mode -- see trace_parser.py -- and must never be conflated
    # with a researcher *choosing* not to search.
    search_tool_error_count: int = 0

    @property
    def first_search_step(self) -> Optional[int]:
        """Informational only (min real step among this episode's executed searches).

        NOT used to determine the post-branch decision boundary -- see
        `_branch_decision_index` for why decision.step and search-event.step
        are not directly comparable near the branch point.
        """
        if not self.search_events:
            return None
        return min(e.step for e in self.search_events)

    @property
    def _branch_decision_index(self) -> Optional[int]:
        """Index within `self.decisions` of the decision that issued the branch-defining
        first search.

        `decisions` are chronologically ordered among themselves (each
        `researcher` node invocation is graph-sequenced after the previous
        one's tool execution), so this is robust even though a decision's
        raw `step` and a search event's raw `step` are not directly
        comparable near the branch point (searches are appended to
        `trace["events"]` in their own trailing counter -- see
        trace_parser module docstring). Requires at least one search to have
        actually executed in this episode (`search_events` non-empty);
        otherwise there is no shared first-search observation to branch from.
        """
        from analysis import config

        if not self.search_events:
            return None
        for idx, decision in enumerate(self.decisions):
            if config.SEARCH_TOOL_NAME in decision.tool_call_names:
                return idx
            if decision.triggered_forced_search:
                # This decision itself never issued a real search tool call
                # (has_tool_calls is genuinely empty), but its emptiness is
                # exactly what triggered ODR's force_benchmark_search safety
                # net to execute a search on its behalf -- it plays the same
                # structural role as if it HAD issued the search itself, so
                # it is the branch decision.
                return idx
        return None

    @property
    def branch_reached(self) -> bool:
        """Whether this episode ever produced a first-search observation."""
        return self._branch_decision_index is not None

    @property
    def search_attempted_but_failed(self) -> bool:
        """True when the episode never got a search observation solely because every
        search-tool invocation errored out (e.g. a Llama tool-call arg
        serialization failure), as opposed to the researcher never
        attempting a search at all. Must be reported as a technical error,
        never as an intentional no-search decision.
        """
        return not self.branch_reached and self.search_tool_error_count > 0

    @property
    def post_branch_decisions(self) -> list[ResearcherDecision]:
        """Researcher decisions strictly after the first-search observation, in order."""
        idx = self._branch_decision_index
        if idx is None:
            return []
        return self.decisions[idx + 1 :]

    @property
    def terminal_post_branch_index(self) -> Optional[int]:
        """1-based post-branch step index (T_i) of the first terminal decision, or None."""
        for idx, decision in enumerate(self.post_branch_decisions, start=1):
            if not decision.parse_ok:
                continue
            if decision.is_research_complete or decision.is_no_tool_exit:
                return idx
        return None

    @property
    def terminal_route(self) -> Optional[str]:
        from analysis import config

        for decision in self.post_branch_decisions:
            if not decision.parse_ok:
                continue
            if decision.is_research_complete:
                return config.ROUTE_RESEARCH_COMPLETE
            if decision.is_no_tool_exit:
                return config.ROUTE_NO_TOOL_EXIT
        return None

    @property
    def terminal_decision_is_imitated_no_tool_call(self) -> bool:
        """True iff `terminal_route` above resolved to ROUTE_NO_TOOL_EXIT via
        a decision whose empty tool_calls is a gemma imitation artifact (see
        ResearcherDecision.reconstructed_from_imitated_history_text), not
        genuine evidence the researcher decided to stop. Checked by
        canonical_completion.classify_local_terminal_route to route such a
        row to TOOL_OR_PARSER_FAILURE instead of NO_TOOL_DIRECT_EXIT."""
        for decision in self.post_branch_decisions:
            if not decision.parse_ok:
                continue
            if decision.is_research_complete:
                return False  # a real ResearchComplete resolved first
            if decision.is_no_tool_exit:
                return decision.reconstructed_from_imitated_history_text
        return False

    @property
    def overall_terminal_decision_index(self) -> Optional[int]:
        """Index (0-based, over ALL decisions incl. the pre-branch one) of this
        episode's terminal decision, regardless of whether a search ever
        happened first. Used for whole-episode termination-route composition
        (Section 9); NOT the PCR step index (see `terminal_post_branch_index`,
        which is scoped to post-branch decisions only, per the PCR definition).
        """
        for idx, decision in enumerate(self.decisions):
            if not decision.parse_ok:
                continue
            if decision.is_research_complete or decision.is_no_tool_exit:
                return idx
        return None

    @property
    def overall_terminal_route(self) -> Optional[str]:
        """Whole-episode terminal route (RC / no-tool-exit), independent of branching.

        None means no terminal decision was observed anywhere in the
        recorded episode (a genuine "censored / not observed in this trace"
        case -- e.g. an event-logging gap or the episode still being open
        when the trace ends), which must be reported as such, not guessed.
        """
        from analysis import config

        idx = self.overall_terminal_decision_index
        if idx is None:
            return None
        decision = self.decisions[idx]
        if decision.is_research_complete:
            return config.ROUTE_RESEARCH_COMPLETE
        return config.ROUTE_NO_TOOL_EXIT

    @property
    def post_injection_search_count(self) -> int:
        """Number of *additional* executed searches after the branch-defining first search.

        `search_events` is appended in true chronological order among
        themselves (see trace_parser module docstring), so this is simply
        "every executed search after the first one" -- no step comparison
        against decisions is needed or valid here.
        """
        return max(0, len(self.search_events) - 1)

    @property
    def any_decision_parse_failed(self) -> bool:
        return any(not d.parse_ok for d in self.decisions)


@dataclass
class ParsedTrace:
    model: str
    dataset: str
    condition: str
    example_id: str
    trace_path: str
    schema_version: Optional[str]

    episodes: list[ResearcherEpisode]

    reached_final_report_generation: bool
    final_report: str
    extracted_answer: str

    # Top-level, dataset-native evidence fields as computed by the (validated,
    # read-only-imported) production runner -- see
    # attack/runtime/runner.py:_coverage_summary_from_search_events.
    # NOTE: "*_at_termination" here is a WHOLE-RUN cumulative measure (spans
    # any supervisor redelegation), not the first attacked episode's raw
    # terminal evidence -- see evidence.py for the boundary-2 recomputation.
    coverage_fraction_at_injection: Optional[float]
    coverage_fraction_at_termination: Optional[float]
    dependencies_seen_at_injection: tuple[str, ...]
    dependencies_seen_at_termination: tuple[str, ...]
    missing_dependency_ids_at_injection: tuple[str, ...]
    missing_dependency_ids_at_termination: tuple[str, ...]
    required_dependency_count: int
    required_paragraph_ids: tuple[str, ...]
    retrieved_paragraph_ids_at_termination: tuple[str, ...]
    hop_count: int

    parse_errors: list[str] = field(default_factory=list)

    @property
    def parse_ok(self) -> bool:
        return not self.parse_errors

    @property
    def first_episode(self) -> Optional[ResearcherEpisode]:
        return self.episodes[0] if self.episodes else None

    @property
    def supervisor_redelegation_count(self) -> int:
        return max(0, len(self.episodes) - 1)

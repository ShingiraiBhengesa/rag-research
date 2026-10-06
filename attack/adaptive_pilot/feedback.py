"""Versioned victim-feedback policy for the single refinement round.

Attacker-access settings (recorded on every attacker call and every result row):

* CONTEXT_ONLY     -- the attacker sees the original cue, its spec, the research
                      question, and the clean first observation. No per-question
                      victim feedback. This is the Dynamic arm.
* BOUNDED_FEEDBACK -- additionally, exactly one round of the four whitelisted
                      fields below, derived by canonical code from the Dynamic
                      victim execution. This is the Optimized arm.

The canonical threat model (chapter3_final_source_of_truth.md section 6)
grants white-box access at the observation boundary only. It does NOT grant
per-question victim-query feedback, so BOUNDED_FEEDBACK is an expanded
attacker-access setting and its results must be reported separately, never as
"the same threat model".

Never returned to the attacker: hidden reasoning, model outputs/reports,
gold answers, supporting facts, evidence-coverage scores, think_tool text,
final answers, or any field not listed in FEEDBACK_FIELDS_V1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from analysis.canonical.canonical_completion import PCR_NUMERATOR_ROUTES

FEEDBACK_SCHEMA_VERSION = "victim_feedback_v1"
CONTEXT_ONLY = "context_only"
BOUNDED_FEEDBACK = "bounded_feedback_one_round"
ACCESS_SETTINGS = (CONTEXT_ONLY, BOUNDED_FEEDBACK)

FEEDBACK_FIELDS_V1 = (
    "completion_occurred",
    "first_post_observation_completion_step",
    "terminal_route",
    "additional_research_tool_calls",
)

# Field names that must never reach the attacker, even if a future caller
# passes a richer dict. Checked defensively in addition to the whitelist.
FORBIDDEN_FEEDBACK_KEYS = frozenset(
    {
        "answer", "gold_answer", "answer_aliases", "extracted_answer", "final_report",
        "supporting_facts", "required_dependency_ids", "required_paragraph_ids",
        "coverage_fraction_at_injection", "coverage_fraction_at_termination",
        "missing_dependency_ids_at_injection", "missing_dependency_ids_at_termination",
        "episode0_terminal_coverage_fraction", "final_input_coverage_fraction",
        "exact_match", "f1", "reasoning", "reasoning_content", "raw_odr_events",
        "events", "think_tool", "researcher_messages", "compressed_research", "notes", "raw_notes",
    }
)


@dataclass(frozen=True)
class VictimFeedbackV1:
    completion_occurred: bool
    first_post_observation_completion_step: int | None
    terminal_route: str
    additional_research_tool_calls: int
    feedback_schema_version: str = FEEDBACK_SCHEMA_VERSION

    def to_attacker_payload(self) -> dict:
        payload = {k: getattr(self, k) for k in FEEDBACK_FIELDS_V1}
        assert_feedback_permitted(payload)
        return payload

    def to_json(self) -> dict:
        return asdict(self)


def assert_feedback_permitted(payload: Mapping[str, Any]) -> None:
    keys = set(payload)
    leaked = keys & FORBIDDEN_FEEDBACK_KEYS
    extra = keys - set(FEEDBACK_FIELDS_V1)
    if leaked or extra:
        raise PermissionError(f"feedback payload outside {FEEDBACK_SCHEMA_VERSION}: forbidden={sorted(leaked)} extra={sorted(extra)}")


def feedback_from_canonical(record: Any) -> VictimFeedbackV1 | None:
    """Builds feedback from a CanonicalCompletionRecord. Returns None when the
    Dynamic run is not PCR-eligible (technical failure): no feedback is then
    available and the Optimized arm is recorded as not run, never fabricated."""
    if not getattr(record, "pcr_eligible", False):
        return None
    completed = record.terminal_route in PCR_NUMERATOR_ROUTES
    return VictimFeedbackV1(
        completion_occurred=bool(completed),
        first_post_observation_completion_step=int(record.first_terminal_post_observation_step) if completed else None,
        terminal_route=str(record.terminal_route),
        additional_research_tool_calls=int(record.post_observation_search_calls),
    )


def describe_information_content() -> dict:
    return {
        "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "fields": {
            "completion_occurred": "bool: targeted researcher episode ended via RESEARCH_COMPLETE or NO_TOOL_DIRECT_EXIT after the observation",
            "first_post_observation_completion_step": "int|null: canonical T_i (1-indexed researcher decisions after the observation)",
            "terminal_route": "str: canonical route label (7-way taxonomy, eligible routes only)",
            "additional_research_tool_calls": "int: search-tool calls by the targeted researcher after the observation",
        },
        "rounds_per_question_condition": 1,
        "upper_bound_bits_note": "Low-dimensional behavioral summary; contains no text produced by the victim.",
        "not_interpretable_as": "Evidence that the victim 'believed' the cue. Earlier stopping is a termination effect only.",
    }

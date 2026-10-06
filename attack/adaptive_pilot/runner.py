"""Four-arm adaptive-cue pilot runner with the single-refinement optimization.

Arms, per (victim_model, dataset, example_id, condition):
  Control    -- canonical NaturalClean, replayed from the shared anchor.
  Static     -- canonical condition suffix, replayed from the shared anchor.
  Dynamic    -- one attacker-generated contextual suffix, no victim feedback.
                This IS attempt one of the optimization procedure.
  Optimized  -- the Dynamic suffix revised once using the whitelisted feedback
                from the Dynamic run (bounded-feedback access only).

Replay faithfulness: the pilot does NOT restore exact victim state. It reuses
the canonical paired-replay mechanism (attack/runtime/runner.py), which pins
the research brief, the ConductResearch topic, and the first search queries
captured from an unforced NaturalClean "anchor-source" run, then re-runs each
arm from scratch with those pinned. Because BM25 is deterministic, this
reproduces the same clean first observation (verified by
base_tool_message_hash / shared_prefix_hash equality across arms, recorded per
cell). This is documented as faithful replay, never as exact state
restoration. Each arm is a fresh victim conversation (its own MemorySaver and
run_id); no arm's conversation is carried into another.

Selection rule: the revised (Optimized) suffix is the final payload. Arms are
never chosen by observed outcome.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from attack.adaptive_pilot.attacker import AttackerCall, AttackerConfig, generate_suffix, refine_suffix
from attack.adaptive_pilot.cue_specs import CueSpec, get_spec
from attack.adaptive_pilot.feedback import (
    BOUNDED_FEEDBACK,
    CONTEXT_ONLY,
    VictimFeedbackV1,
    feedback_from_canonical,
)
from attack.adaptive_pilot.validation import (
    TokenCounter,
    build_observation,
    expected_tool_message_hash,
    verify_observation,
)
from attack.runtime.cues import PILOT_SUFFIX_OVERRIDE_KEY, suffix_for_condition
from attack.runtime.hashing import hash_text, stable_hash
from attack.runtime.runner import (
    V3ModelConfig,
    V3RunRequest,
    _build_replay_anchor,
    run_single_condition,
)

ARMS = ("control", "static", "dynamic", "optimized")
CONTROL_CONDITION = "NaturalClean"


def pilot_run_id(*, victim_model: str, dataset: str, example_id: str, condition: str, arm: str, access: str) -> str:
    return stable_hash(
        {
            "schema": "adaptive_cue_pilot_run_id_v1",
            "victim_model": victim_model,
            "dataset": dataset,
            "example_id": example_id,
            "condition": condition,
            "arm": arm,
            "access": access,
        },
        length=None,
    )


@dataclass
class ArmResult:
    arm: str
    condition: str
    run_id: str
    suffix: str
    suffix_source: str          # "none" | "canonical_static" | "attacker_generation" | "attacker_refinement"
    suffix_hash: str
    trace_path: str
    base_tool_message_hash: Optional[str]
    shared_prefix_hash: Optional[str]
    attacked_observation: Optional[bool]
    observation_check: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    error: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


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
    from attack.runtime.benchmark_search import run_benchmark_search

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


@dataclass
class PilotCellConfig:
    victim_model_key: str
    victim: V3ModelConfig
    attacker: AttackerConfig
    dataset: str
    example_id: str
    question: str
    condition: str
    corpus_path: str
    evidence_path: Optional[str]
    output_dir: str
    access_setting: str = BOUNDED_FEEDBACK
    suffix_token_limit: Optional[int] = None
    top_k: int = 3
    emergency_max_react_tool_calls: int = 30
    max_researcher_iterations: int = 6


def run_pilot_cell(cfg: PilotCellConfig, token_counter: TokenCounter | None) -> dict:
    """Runs all four arms for one (victim_model, dataset, example_id, condition).

    Returns a single JSON-serializable record. Never raises on an attack
    failing; raises only on infrastructure faults the caller should retry.
    """
    spec: CueSpec = get_spec(cfg.condition, suffix_token_limit=cfg.suffix_token_limit)
    out = Path(cfg.output_dir)
    base_kwargs = dict(
        dataset=cfg.dataset, example_id=cfg.example_id, question=cfg.question,
        corpus_path=cfg.corpus_path, evidence_path=cfg.evidence_path, output_dir=cfg.output_dir,
        model=cfg.victim, top_k=cfg.top_k, emergency_max_react_tool_calls=cfg.emergency_max_react_tool_calls,
        max_researcher_iterations=cfg.max_researcher_iterations,
    )

    started = time.perf_counter()

    # 1) Anchor-source run (unforced NaturalClean) to capture brief/topic/queries.
    anchor_src_id = pilot_run_id(
        victim_model=cfg.victim_model_key, dataset=cfg.dataset, example_id=cfg.example_id,
        condition=cfg.condition, arm="anchor_source", access=cfg.access_setting,
    )
    anchor_trace = run_single_condition(
        _victim_request(condition=CONTROL_CONDITION, run_id=anchor_src_id, replay_anchor=None, cue_overrides={}, **base_kwargs)
    )
    anchor = _build_replay_anchor(anchor_trace)

    # 2) Clean observation text for the attacker, verified against the anchor hash.
    clean_obs = _clean_observation_text(
        corpus_path=cfg.corpus_path, evidence_path=cfg.evidence_path, dataset=cfg.dataset,
        example_id=cfg.example_id, queries=anchor["first_search_queries"], output_dir=cfg.output_dir, top_k=cfg.top_k,
    )
    clean_obs_hash = hash_text(clean_obs, length=32)
    clean_obs_matches_anchor = clean_obs_hash == anchor.get("source_base_tool_message_hash")

    arms: dict[str, ArmResult] = {}

    def run_arm(arm: str, condition: str, suffix: str | None, suffix_source: str) -> ArmResult:
        cue_overrides = {} if suffix is None else {PILOT_SUFFIX_OVERRIDE_KEY: suffix}
        rid = pilot_run_id(
            victim_model=cfg.victim_model_key, dataset=cfg.dataset, example_id=cfg.example_id,
            condition=condition, arm=arm, access=cfg.access_setting,
        )
        t0 = time.perf_counter()
        try:
            trace = run_single_condition(
                _victim_request(condition=condition, run_id=rid, replay_anchor=anchor, cue_overrides=cue_overrides, **base_kwargs)
            )
        except Exception as exc:  # infrastructure fault for this arm
            return ArmResult(arm=arm, condition=condition, run_id=rid, suffix=suffix or "",
                             suffix_source=suffix_source, suffix_hash=hash_text(suffix, length=None) if suffix else "",
                             trace_path="", base_tool_message_hash=None, shared_prefix_hash=None,
                             attacked_observation=None, runtime_seconds=time.perf_counter() - t0, error=repr(exc))
        ev = _first_search_event(trace)
        checks: list[str] = []
        if suffix:
            # Verify the victim-visible observation is exactly clean + SEP + suffix.
            if ev.get("tool_message_hash") != expected_tool_message_hash(clean_obs, suffix):
                checks.append("tool_message_hash_mismatch_vs_expected_append")
            if ev.get("base_tool_message_hash") != clean_obs_hash:
                checks.append("clean_prefix_hash_diverged_from_anchor")
        return ArmResult(
            arm=arm, condition=condition, run_id=rid, suffix=suffix or "", suffix_source=suffix_source,
            suffix_hash=ev.get("suffix_hash") or (hash_text(suffix, length=None) if suffix else ""),
            trace_path=str(trace.get("trace_path") or ""),
            base_tool_message_hash=ev.get("base_tool_message_hash"),
            shared_prefix_hash=ev.get("shared_prefix_hash"),
            attacked_observation=ev.get("attacked_observation"),
            observation_check=checks, runtime_seconds=time.perf_counter() - t0,
        )

    # 3) Control (replayed NaturalClean) and Static (replayed canonical suffix).
    arms["control"] = run_arm("control", CONTROL_CONDITION, None, "none")
    static_suffix = suffix_for_condition(cfg.condition).suffix
    arms["static"] = run_arm("static", cfg.condition, static_suffix, "canonical_static")

    # 4) Dynamic = optimization attempt one (no victim feedback).
    gen_call: AttackerCall = generate_suffix(cfg.attacker, spec, cfg.question, clean_obs, token_counter)
    if gen_call.accepted:
        arms["dynamic"] = run_arm("dynamic", cfg.condition, gen_call.suffix, "attacker_generation")
    else:
        arms["dynamic"] = ArmResult(
            arm="dynamic", condition=cfg.condition,
            run_id=pilot_run_id(victim_model=cfg.victim_model_key, dataset=cfg.dataset, example_id=cfg.example_id,
                                condition=cfg.condition, arm="dynamic", access=cfg.access_setting),
            suffix="", suffix_source="attacker_generation", suffix_hash="", trace_path="",
            base_tool_message_hash=None, shared_prefix_hash=None, attacked_observation=None,
            error="dynamic_generation_failed_validation",
        )

    # 5) Feedback from the Dynamic run (bounded-feedback access only).
    feedback: VictimFeedbackV1 | None = None
    feedback_available = False
    refine_call: AttackerCall | None = None
    if cfg.access_setting == BOUNDED_FEEDBACK and arms["dynamic"].trace_path and not arms["dynamic"].error:
        from analysis.completion_inventory import RowRecord
        from analysis.canonical.canonical_completion import build_canonical_record

        dyn_row = RowRecord(
            model=cfg.victim_model_key, dataset=cfg.dataset, condition=cfg.condition, shard_index=0,
            example_id=cfg.example_id, final_status="ok", error="", trace_path=arms["dynamic"].trace_path,
            trace_exists=Path(arms["dynamic"].trace_path).exists(), valid_answer=None, row_index_in_shard=0,
            run_id=arms["dynamic"].run_id,
        )
        feedback = feedback_from_canonical(build_canonical_record(dyn_row))
        feedback_available = feedback is not None

    # 6) Optimized = single revision (bounded-feedback only; final payload by rule).
    if cfg.access_setting == BOUNDED_FEEDBACK and feedback is not None and gen_call.accepted:
        refine_call = refine_suffix(
            cfg.attacker, spec, cfg.question, clean_obs, gen_call.suffix, feedback.to_attacker_payload(), token_counter
        )
        if refine_call.accepted:
            arms["optimized"] = run_arm("optimized", cfg.condition, refine_call.suffix, "attacker_refinement")
        else:
            arms["optimized"] = ArmResult(
                arm="optimized", condition=cfg.condition,
                run_id=pilot_run_id(victim_model=cfg.victim_model_key, dataset=cfg.dataset, example_id=cfg.example_id,
                                    condition=cfg.condition, arm="optimized", access=cfg.access_setting),
                suffix="", suffix_source="attacker_refinement", suffix_hash="", trace_path="",
                base_tool_message_hash=None, shared_prefix_hash=None, attacked_observation=None,
                error="refinement_failed_validation",
            )

    # Shared-prefix agreement across all executed arms (faithful-replay check).
    prefix_hashes = {a.base_tool_message_hash for a in arms.values() if a.base_tool_message_hash}
    shared_prefix_ok = len(prefix_hashes) == 1 and clean_obs_matches_anchor

    record = {
        "schema_version": "adaptive_cue_pilot_cell_v1",
        "pilot_version": "adaptive_cue_pilot_v1",
        "access_setting": cfg.access_setting,
        "victim_model": cfg.victim_model_key,
        "victim_model_id": cfg.victim.model_name,
        "attacker": cfg.attacker.fingerprint(),
        "attacker_is_victim_family": cfg.victim.model_name == cfg.attacker.model_id,
        "dataset": cfg.dataset,
        "example_id": cfg.example_id,
        "condition": cfg.condition,
        "suffix_token_limit": spec.suffix_token_limit,
        "clean_observation_hash": clean_obs_hash,
        "clean_observation_matches_anchor": clean_obs_matches_anchor,
        "shared_prefix_ok": shared_prefix_ok,
        "anchor": {k: anchor.get(k) for k in ("research_brief", "research_topic", "first_search_queries",
                                              "source_base_tool_message_hash", "source_shared_prefix_hash")},
        "arms": {name: a.to_json() for name, a in arms.items()},
        "attacker_generation": gen_call.to_json(),
        "attacker_refinement": refine_call.to_json() if refine_call else None,
        "feedback_available": feedback_available,
        "feedback_payload": feedback.to_attacker_payload() if feedback is not None else None,
        "feedback_full": feedback.to_json() if feedback is not None else None,
        "cell_runtime_seconds": time.perf_counter() - started,
    }
    return record

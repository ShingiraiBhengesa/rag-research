"""Victim-side phases: anchor capture and the six replayed arms.

The victim pipeline is the canonical one, unchanged
(attack/runtime/runner.py::run_single_condition with faithful replay). The
anchor-source run is an unforced NaturalClean run that pins the research
brief, topic, and first search queries; every arm replays from it, so all six
arms see the same clean first observation d1, verified per arm by hash.

Each arm writes into its own output_dir. The canonical runner names the trace
path by (dataset, example_id, condition) only, and three arm pairs share a
runtime condition, so a shared output_dir would let one arm overwrite
another's trace.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from attack.adaptive_pilot.runner import _clean_observation_text, _first_search_event, _victim_request
from attack.adaptive_pilot.validation import expected_tool_message_hash
from attack.dynamic_bridge_pilot import config as C
from attack.runtime.cues import PILOT_SUFFIX_OVERRIDE_KEY, suffix_for_condition
from attack.runtime.hashing import hash_text, stable_hash
from attack.runtime.runner import V3ModelConfig, _build_replay_anchor, run_single_condition


def _run_victim(request) -> dict:
    """One fresh victim run. The canonical runner is async (production calls it via
    asyncio.run). The canonical search tool keeps per-process counters keyed by
    run_id that gate the one-shot injection; clear them first so a technical retry
    of the same run_id is injected exactly like the first attempt."""
    from attack.runtime import benchmark_search as bs

    rid = request.run_id
    bs._RUN_INJECTION_COUNTS.pop(rid, None)
    for key in [k for k in bs._RUN_SEARCH_COUNTS if k[0] == rid]:
        bs._RUN_SEARCH_COUNTS.pop(key, None)
    return asyncio.run(run_single_condition(request))


def unit_key(*parts: str) -> str:
    return "|".join(parts)


def run_id_for(victim: str, dataset: str, example_id: str, arm: str) -> str:
    return stable_hash({"schema": C.PILOT_VERSION + "_run_id", "victim": victim, "dataset": dataset,
                        "example_id": example_id, "arm": arm}, length=None)


@dataclass
class VictimContext:
    victim_key: str
    model: V3ModelConfig
    traces_root: Path

    def request_kwargs(self, *, dataset: str, q: Mapping[str, Any], paths: Mapping[str, Path], arm: str) -> dict:
        return dict(
            dataset=dataset, example_id=q["example_id"], question=q["question"],
            corpus_path=paths["corpus"], evidence_path=paths["evidence"],
            output_dir=self.traces_root / self.victim_key / arm, model=self.model, top_k=C.TOP_K,
            emergency_max_react_tool_calls=C.EMERGENCY_MAX_REACT_TOOL_CALLS,
            max_researcher_iterations=C.MAX_RESEARCHER_ITERATIONS,
        )


def run_anchor(ctx: VictimContext, *, dataset: str, q: Mapping[str, Any], paths: Mapping[str, Path]) -> dict:
    t0 = time.perf_counter()
    kw = ctx.request_kwargs(dataset=dataset, q=q, paths=paths, arm="anchor_source")
    rid = run_id_for(ctx.victim_key, dataset, q["example_id"], "anchor_source")
    trace = _run_victim(_victim_request(condition="NaturalClean", run_id=rid, replay_anchor=None,
                                        cue_overrides={}, **kw))
    anchor = _build_replay_anchor(trace)
    clean_obs = _clean_observation_text(
        corpus_path=paths["corpus"], evidence_path=paths["evidence"], dataset=dataset, example_id=q["example_id"],
        queries=anchor["first_search_queries"], output_dir=kw["output_dir"], top_k=C.TOP_K)
    obs_hash = hash_text(clean_obs, length=32)
    return {
        "victim": ctx.victim_key, "dataset": dataset, "example_id": q["example_id"], "question": q["question"],
        "anchor": anchor, "anchor_trace_path": str(trace.get("trace_path") or ""), "anchor_run_id": rid,
        "clean_observation": clean_obs, "clean_observation_hash": obs_hash,
        "clean_observation_matches_anchor": obs_hash == anchor.get("source_base_tool_message_hash"),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }


@dataclass
class ArmPlan:
    arm: str
    runtime_condition: str
    suffix: Optional[str]          # None = canonical resolution (neutral / fixed arms)
    suffix_source: str
    runnable: bool
    skip_reason: str = ""
    provenance: dict = field(default_factory=dict)


def plan_arms(bridge_rec: Mapping[str, Any] | None, selections: Mapping[str, Mapping[str, Any]]) -> list[ArmPlan]:
    """Resolve every configured arm for one (victim, question).

    selections[arm] = {"selected": variant dict or None, "reason": str} for the
    dynamic arms (written by the select phase). Neutral and fixed arms always run.
    """
    plans: list[ArmPlan] = []
    bridge_ok = bool(bridge_rec and bridge_rec.get("status") == "OK")
    for arm, spec in C.ARMS.items():
        cond = spec["runtime_condition"]
        src = spec["suffix_source"]
        if src in ("none", "canonical_fixed"):
            plans.append(ArmPlan(arm, cond, None, src, True))
            continue
        if not bridge_ok:
            plans.append(ArmPlan(arm, cond, None, src, False,
                                 skip_reason=(bridge_rec or {}).get("status", "NO_BRIDGE_RECORD")))
            continue
        sel = selections.get(arm) or {}
        u = sel.get("selected")
        if u is not None:
            plans.append(ArmPlan(arm, cond, u["cue"], src, True,
                                 provenance={"unit": spec["unit"], "selection": spec["selection"],
                                             "variant_index": u["index"], "seed": u["seed"]}))
        else:
            plans.append(ArmPlan(arm, cond, None, src, False, skip_reason=sel.get("reason") or "NO_SELECTION"))
    return plans


def run_arm(ctx: VictimContext, plan: ArmPlan, *, dataset: str, q: Mapping[str, Any], paths: Mapping[str, Path],
            anchor_rec: Mapping[str, Any]) -> dict:
    base = {"victim": ctx.victim_key, "dataset": dataset, "example_id": q["example_id"], "arm": plan.arm,
            "runtime_condition": plan.runtime_condition, "suffix_source": plan.suffix_source,
            "provenance": plan.provenance}
    if not plan.runnable:
        return {**base, "status": "not_run", "skip_reason": plan.skip_reason, "trace_path": "", "suffix": ""}
    overrides = {} if plan.suffix is None else {PILOT_SUFFIX_OVERRIDE_KEY: plan.suffix}
    effective_suffix = suffix_for_condition(plan.runtime_condition, overrides).suffix
    rid = run_id_for(ctx.victim_key, dataset, q["example_id"], plan.arm)
    kw = ctx.request_kwargs(dataset=dataset, q=q, paths=paths, arm=plan.arm)
    t0 = time.perf_counter()
    trace = _run_victim(_victim_request(condition=plan.runtime_condition, run_id=rid,
                                        replay_anchor=anchor_rec["anchor"], cue_overrides=overrides, **kw))
    ev = _first_search_event(trace)
    clean = anchor_rec["clean_observation"]
    checks = []
    if ev.get("base_tool_message_hash") != anchor_rec["clean_observation_hash"]:
        checks.append("clean_prefix_diverged_from_anchor")
    if ev.get("tool_message_hash") != expected_tool_message_hash(clean, effective_suffix or None):
        checks.append("victim_observation_not_exact_clean_plus_suffix")
    return {**base, "status": "ran", "run_id": rid, "suffix": effective_suffix,
            "suffix_hash": hash_text(effective_suffix, length=None) if effective_suffix else "",
            "trace_path": str(trace.get("trace_path") or ""),
            "base_tool_message_hash": ev.get("base_tool_message_hash"),
            "tool_message_hash": ev.get("tool_message_hash"),
            "attacked_observation": ev.get("attacked_observation"),
            "injection_checks": checks, "injection_ok": not checks,
            "runtime_seconds": round(time.perf_counter() - t0, 2)}

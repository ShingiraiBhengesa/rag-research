"""Unit + regression tests for canonical_completion.py.

Run with:
    cd <repo root>
    python -m pytest tests/evaluation/test_completion.py -v

or, if pytest is unavailable in the active environment:
    python tests/evaluation/test_completion.py

Tests operate on synthetic ResearcherEpisode/ResearcherDecision/SearchEvent
objects (no production trace files touched) for items 1-17, and on one real,
already-completed, read-only production trace file for item 20 (a direct
regression check against the pre-existing pcr.py's own computation on that
exact file -- see build_canonical_record vs prematurestop.evaluation.pcr).
No file under results/ is modified by any test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `attack.*`/`analysis.*` imports

from prematurestop.evaluation import config
from prematurestop.evaluation.trace_schema import ResearcherDecision, ResearcherEpisode, SearchEvent
from prematurestop.evaluation import completion as cc
from prematurestop.paths import RESULTS_E1_CANONICAL


def _decision(step: int, tool_names: list[str], parse_ok: bool = True) -> ResearcherDecision:
    is_rc = "ResearchComplete" in tool_names
    is_no_tool = len(tool_names) == 0
    return ResearcherDecision(
        step=step,
        subgraph_id="ep0",
        tool_call_names=tool_names,
        tool_calls_raw=[{"name": n} for n in tool_names],
        has_tool_calls=bool(tool_names),
        is_research_complete=is_rc,
        is_no_tool_exit=is_no_tool,
        parse_ok=parse_ok,
    )


def _search_event(step: int, attacked: bool = False) -> SearchEvent:
    return SearchEvent(
        step=step,
        subgraph_id="ep0",
        supervisor_delegation_index=0,
        attacked_observation=attacked,
        coverage_fraction_after_observation=0.5,
        dependencies_seen_after_observation=(),
        missing_dependency_ids_after_observation=(),
        required_dependency_ids=(),
        required_paragraph_ids=(),
        retrieved_paragraph_ids=(),
        normalized_queries=("q",),
        raw={},
    )


def _episode(decisions: list[ResearcherDecision], search_events: list[SearchEvent], search_tool_error_count: int = 0) -> ResearcherEpisode:
    return ResearcherEpisode(
        subgraph_id="ep0",
        episode_order=0,
        supervisor_iter=1,
        decisions=decisions,
        search_events=search_events,
        search_tool_error_count=search_tool_error_count,
    )


class TestStepAndRouteBasics(unittest.TestCase):
    def test_1_immediate_research_complete_gives_t1(self):
        # decision0: search (branch); decision1: ResearchComplete (T_i=1)
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, ["ResearchComplete"])],
            [_search_event(0)],
        )
        route, reason = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.RESEARCH_COMPLETE)
        self.assertEqual(reason, "")
        self.assertEqual(cc.first_local_completion_step(ep), 1)

    def test_2_immediate_no_tool_exit_gives_t1(self):
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, [])],
            [_search_event(0)],
        )
        route, reason = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.NO_TOOL_DIRECT_EXIT)
        self.assertEqual(reason, "")
        self.assertEqual(cc.first_local_completion_step(ep), 1)

    def test_3_search_then_research_complete_gives_later_step(self):
        # branch decision, then think, then search, then ResearchComplete => T_i=3
        ep = _episode(
            [
                _decision(0, [config.SEARCH_TOOL_NAME]),
                _decision(1, [config.THINK_TOOL_NAME]),
                _decision(2, [config.SEARCH_TOOL_NAME]),
                _decision(3, ["ResearchComplete"]),
            ],
            [_search_event(0), _search_event(10)],
        )
        self.assertEqual(cc.first_local_completion_step(ep), 3)
        route, _ = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.RESEARCH_COMPLETE)

    def test_4_search_then_no_tool_exit_gives_later_step(self):
        ep = _episode(
            [
                _decision(0, [config.SEARCH_TOOL_NAME]),
                _decision(1, [config.SEARCH_TOOL_NAME]),
                _decision(2, []),
            ],
            [_search_event(0), _search_event(5)],
        )
        self.assertEqual(cc.first_local_completion_step(ep), 2)
        route, _ = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.NO_TOOL_DIRECT_EXIT)

    def test_5_think_tool_counts_as_one_decision_step(self):
        # observation -> think_tool -> search -> ResearchComplete
        # decisions after branch: [think_tool](step1), [search](step2), [RC](step3)
        ep = _episode(
            [
                _decision(0, [config.SEARCH_TOOL_NAME]),
                _decision(1, [config.THINK_TOOL_NAME]),
                _decision(2, [config.SEARCH_TOOL_NAME]),
                _decision(3, ["ResearchComplete"]),
            ],
            [_search_event(0), _search_event(9)],
        )
        traj = cc.extract_post_observation_trajectory(ep)
        self.assertEqual(traj["post_observation_researcher_decisions"], 3)  # think, search, RC
        self.assertEqual(traj["post_observation_think_calls"], 1)
        self.assertEqual(traj["post_observation_search_calls"], 1)
        self.assertEqual(cc.first_local_completion_step(ep), 3)  # RC is the 3rd post-branch decision

    def test_6_run_cannot_count_in_both_route_categories(self):
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, ["ResearchComplete"])],
            [_search_event(0)],
        )
        route, _ = cc.classify_local_terminal_route(ep)
        self.assertIn(route, cc.CANONICAL_ROUTES)
        self.assertEqual(sum(1 for r in cc.CANONICAL_ROUTES if r == route), 1)

    def test_7_union_pcr_equals_sum_of_route_specific_rates(self):
        records = []
        # 3 RC at steps 1,2,3 and 2 no-tool at steps 1,4 => 5 eligible
        for i, step in enumerate([1, 2, 3]):
            records.append(_fake_record(f"rc{i}", cc.RESEARCH_COMPLETE, step, step, None))
        for i, step in enumerate([1, 4]):
            records.append(_fake_record(f"nt{i}", cc.NO_TOOL_DIRECT_EXIT, step, None, step))
        union = cc.compute_pcr_curve(records, k_max=5)
        route_specific = cc.compute_route_specific_curves(records, k_max=5)
        for u, r in zip(union["per_step"], route_specific["per_step"]):
            self.assertAlmostEqual(u["pcr_at_k"], r["union_pct"])
            self.assertAlmostEqual(r["union_pct"], (r["rc_pct"] or 0) + (r["no_tool_pct"] or 0))

    def test_8_pcr_at_k_is_monotonic(self):
        records = [
            _fake_record("a", cc.RESEARCH_COMPLETE, 3, 3, None),
            _fake_record("b", cc.NO_TOOL_DIRECT_EXIT, 7, None, 7),
            _fake_record("c", cc.MAX_BUDGET_EXIT, None, None, None),
        ]
        curve = cc.compute_pcr_curve(records, k_max=10)
        pcts = [s["pcr_at_k"] for s in curve["per_step"]]
        for a, b in zip(pcts, pcts[1:]):
            self.assertLessEqual(a, b)


class TestAnchorSemantics(unittest.TestCase):
    def test_9_clean_condition_uses_equivalent_first_search_anchor(self):
        # NaturalClean: no injection, but anchor is still "the first executed search"
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, ["ResearchComplete"])],
            [_search_event(0, attacked=False)],
        )
        idx, step, injected = cc._anchor_fields(ep)
        self.assertIsNotNone(idx)
        self.assertFalse(injected)

    def test_10_attack_condition_uses_attacked_observation_anchor(self):
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, ["ResearchComplete"])],
            [_search_event(0, attacked=True)],
        )
        idx, step, injected = cc._anchor_fields(ep)
        self.assertIsNotNone(idx)
        self.assertTrue(injected)

    def test_11_oracle_condition_uses_oracle_observation_anchor(self):
        # Oracle-complete conditions still anchor on "first executed search" --
        # the parser is condition-agnostic; oracle-specific retrieval content
        # doesn't change which decision is the branch decision.
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, [])],
            [_search_event(0, attacked=True)],  # OracleComplete_* conditions can still carry a suffix
        )
        idx, step, injected = cc._anchor_fields(ep)
        self.assertIsNotNone(idx)
        self.assertEqual(cc.first_local_completion_step(ep), 1)


class TestFailureRoutes(unittest.TestCase):
    def test_12_max_budget_exit_not_counted_as_voluntary_completion(self):
        decisions = [_decision(0, [config.SEARCH_TOOL_NAME])]
        # 30 total decisions, none ever resolve to RC/no-tool (all just search again)
        for i in range(1, cc.MAX_REACT_TOOL_CALLS):
            decisions.append(_decision(i, [config.SEARCH_TOOL_NAME]))
        ep = _episode(decisions, [_search_event(0)])
        route, reason = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.MAX_BUDGET_EXIT)
        self.assertNotIn(route, cc.PCR_NUMERATOR_ROUTES)

    def test_13_tool_parser_failure_not_counted_as_no_tool_exit(self):
        ep = _episode([], [], search_tool_error_count=3)
        route, reason = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.TOOL_OR_PARSER_FAILURE)
        self.assertNotEqual(route, cc.NO_TOOL_DIRECT_EXIT)
        self.assertFalse(ep.branch_reached)

    def test_14_trace_logging_failure_not_silently_counted_as_completion(self):
        # branch reached but decisions run out (< budget) with no terminal ever found
        ep = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, [config.THINK_TOOL_NAME])],
            [_search_event(0)],
        )
        route, reason = cc.classify_local_terminal_route(ep)
        self.assertEqual(route, cc.TRACE_LOGGING_FAILURE)
        self.assertNotIn(route, cc.PCR_NUMERATOR_ROUTES)

    def test_15_missing_anchor_excluded_with_explicit_reason(self):
        # No search ever attempted, episode has no terminal either (e.g. logging gap)
        ep = _episode([_decision(0, [config.THINK_TOOL_NAME])], [])
        route, reason = cc.classify_local_terminal_route(ep)
        self.assertFalse(ep.branch_reached)
        self.assertNotEqual(reason, "")


class TestSupervisorAndReportSeparation(unittest.TestCase):
    def test_16_supervisor_redelegation_does_not_change_targeted_episode_t_i(self):
        ep0 = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, ["ResearchComplete"])],
            [_search_event(0)],
        )
        # T_i is computed purely from episode 0's own decisions -- a second
        # (redelegated) episode existing elsewhere in the trace cannot alter it.
        t_i_before = cc.first_local_completion_step(ep0)
        _ep1_unused = _episode(
            [_decision(0, [config.SEARCH_TOOL_NAME]), _decision(1, [config.SEARCH_TOOL_NAME]), _decision(2, ["ResearchComplete"])],
            [_search_event(0), _search_event(1)],
        )
        t_i_after = cc.first_local_completion_step(ep0)
        self.assertEqual(t_i_before, t_i_after)

    def test_17_final_report_generation_never_classified_as_local_completion(self):
        # canonical_completion.py's route vocabulary has no "final report" route
        # at all -- final_report_generation is a top-level-graph node, never a
        # researcher-episode terminal route, by construction of this module's
        # taxonomy (it only classifies ResearcherEpisode objects).
        self.assertNotIn("FINAL_REPORT", cc.CANONICAL_ROUTES)


def _fake_record(example_id, route, t_i, rc_step, nt_step, eligible=True, model="qwen14b", dataset="hotpot", condition="NaturalClean"):
    return cc.CanonicalCompletionRecord(
        model=model, dataset=dataset, condition=condition, example_id=example_id, run_id=example_id,
        trace_path="", researcher_episode_id="ep0", anchor_found=True, anchor_event_index=0, anchor_step=0,
        injection_applied=False, first_terminal_post_observation_step=t_i, terminal_route=route,
        research_complete_step=rc_step, no_tool_direct_exit_step=nt_step,
        post_observation_researcher_decisions=t_i or 0, post_observation_search_calls=0, post_observation_think_calls=0,
        supervisor_redelegated=False, final_status="ok", parse_valid=True,
        pcr_eligible=eligible, pcr_exclusion_reason="" if eligible else "test",
    )


class TestPairing(unittest.TestCase):
    def test_18_paired_comparison_uses_exact_shared_example_ids(self):
        records = [
            _fake_record("ex1", cc.RESEARCH_COMPLETE, 1, 1, None, condition="NaturalClean"),
            _fake_record("ex2", cc.RESEARCH_COMPLETE, 2, 2, None, condition="NaturalClean"),
            _fake_record("ex1", cc.NO_TOOL_DIRECT_EXIT, 1, None, 1, condition="Natural_epistemic_confirmation"),
            # ex2 has no counterpart under the attack condition
            _fake_record("ex3", cc.RESEARCH_COMPLETE, 1, 1, None, condition="Natural_epistemic_confirmation"),
        ]
        pairs = cc.build_strict_pairs(records, "Natural_epistemic_confirmation", "NaturalClean")
        self.assertEqual(len(pairs), 1)
        a, b = pairs[0]
        self.assertEqual(a.example_id, "ex1")
        self.assertEqual(b.example_id, "ex1")

    def test_19_unequal_sample_sizes_do_not_create_unpaired_delta(self):
        records = [
            _fake_record("ex1", cc.RESEARCH_COMPLETE, 1, 1, None, condition="NaturalClean"),
            _fake_record("ex2", cc.RESEARCH_COMPLETE, 2, 2, None, condition="NaturalClean"),
            _fake_record("ex3", cc.RESEARCH_COMPLETE, 1, 1, None, condition="NaturalClean"),
            _fake_record("ex1", cc.RESEARCH_COMPLETE, 1, 1, None, condition="Natural_search_match_cue"),
        ]
        pairs = cc.build_strict_pairs(records, "Natural_search_match_cue", "NaturalClean")
        # Only ex1 has both sides -- raw condition totals (1 vs 3) must never
        # be used as the paired N; paired N is exactly len(pairs).
        self.assertEqual(len(pairs), 1)
        raw_baseline_n = sum(1 for r in records if r.condition == "NaturalClean")
        self.assertNotEqual(len(pairs), raw_baseline_n)


class TestRealFixtureRegression(unittest.TestCase):
    """Item 20: reproduce a known terminal classification from a real,
    already-completed qwen14b production trace, cross-checked against the
    pre-existing pcr.py's own PCRRecord for the identical file. Skips
    gracefully (does not fail the suite) if no qwen14b shard is present in
    this environment, since this is a read-only regression check against
    live production data, not a hermetic unit test.
    """

    def test_20_real_fixture_reproduces_expected_classification(self):
        import json

        from prematurestop.evaluation.completion_inventory import _shard_path

        repo_root = config.REPO_ROOT
        production_root = repo_root / "results"
        shard_path = _shard_path(production_root, "qwen14b", "hotpot", "NaturalClean", 0)
        if not shard_path.exists():
            self.skipTest(f"no production shard at {shard_path}; skipping live-data regression check")

        with open(shard_path, "r", encoding="utf-8") as fh:
            first_ok_row = None
            for line in fh:
                row = json.loads(line)
                if row.get("final_status") == "ok" and row.get("trace_path"):
                    first_ok_row = row
                    break
        if first_ok_row is None:
            self.skipTest("no valid ok row found in first qwen14b/hotpot/NaturalClean shard")

        from prematurestop.evaluation.completion_inventory import RowRecord
        from prematurestop.evaluation.pcr import _parse_one

        normalized_trace_path = config.normalize_stored_trace_path(str(first_ok_row["trace_path"]))
        row = RowRecord(
            model="qwen14b", dataset="hotpot", condition="NaturalClean", shard_index=0,
            example_id=str(first_ok_row["example_id"]), final_status="ok", error="",
            trace_path=normalized_trace_path,
            trace_exists=(repo_root / normalized_trace_path).exists(),
            valid_answer=first_ok_row.get("valid_answer"), row_index_in_shard=0,
            run_id=str(first_ok_row.get("run_id") or ""),
        )
        if not row.trace_exists:
            self.skipTest("referenced trace file does not exist on disk")

        canonical = cc.build_canonical_record(row)
        legacy = _parse_one(row.model, row.dataset, row.condition, row.example_id, row.trace_path)

        # Where the legacy record found an observed terminal route, the
        # canonical route must agree (RC<->RESEARCH_COMPLETE, no-tool<->NO_TOOL_DIRECT_EXIT)
        # and T_i must match exactly -- this module reuses the same parser,
        # so any mismatch here would indicate a real bug in this module.
        if legacy.terminal_route == config.ROUTE_RESEARCH_COMPLETE:
            self.assertEqual(canonical.terminal_route, cc.RESEARCH_COMPLETE)
            self.assertEqual(canonical.first_terminal_post_observation_step, legacy.terminal_post_branch_index)
        elif legacy.terminal_route == config.ROUTE_NO_TOOL_EXIT:
            self.assertEqual(canonical.terminal_route, cc.NO_TOOL_DIRECT_EXIT)
            self.assertEqual(canonical.first_terminal_post_observation_step, legacy.terminal_post_branch_index)


class TestConditionMetadataConsistency(unittest.TestCase):
    """condition_metadata.csv is a hand-curated table (it carries thesis-only
    editorial columns -- display label, control/intervention, primary/secondary
    -- that can't be derived from code) documenting all 12 conditions in one
    place for citation. Its `exact_suffix` column is not hand-curated data,
    though: it must always equal what `prematurestop.cues.fixed.registry.suffix_for_condition`
    actually resolves, or the cited table silently drifts from runtime
    behavior. This test is that guarantee.
    """

    def test_22_condition_metadata_suffix_matches_runtime(self):
        import csv

        from prematurestop.cues.fixed.registry import V3_CONDITIONS, suffix_for_condition

        csv_path = RESULTS_E1_CANONICAL / "condition_metadata.csv"
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = {row["condition_id"]: row["exact_suffix"] for row in csv.DictReader(fh)}

        self.assertEqual(set(rows), set(V3_CONDITIONS))
        for condition in V3_CONDITIONS:
            actual_suffix = suffix_for_condition(condition).suffix
            expected_suffix = rows[condition] if actual_suffix else "(none -- no injection)"
            self.assertEqual(
                actual_suffix or "(none -- no injection)",
                expected_suffix,
                f"condition_metadata.csv exact_suffix for {condition!r} is stale",
            )


if __name__ == "__main__":
    unittest.main()

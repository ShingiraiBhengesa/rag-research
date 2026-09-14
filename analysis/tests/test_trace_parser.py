"""Tests for trace_parser.py, anchored on real production traces that were
manually inspected and hand-verified against the raw JSON before being used
as fixtures (see the module docstrings in trace_parser.py / trace_schema.py
for the reasoning). Raw traces are not published (Section 20), so every test
here skips gracefully when the production data isn't present on disk --
these are regression/verification tests for this machine's data, not
synthetic unit fixtures.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from analysis import config
from analysis.trace_parser import parse_trace_file

QWEN_NO_TOOL_EXIT_TRACE = (
    config.DEFAULT_PRODUCTION_ROOT
    / "traces/qwen14b/branches/multihoprag/mhrag_1024/NaturalNeutral/trace.json"
)
QWEN_REDELEGATION_TRACE = (
    config.DEFAULT_PRODUCTION_ROOT
    / "traces/qwen14b/branches/hotpot/5a84b6e15542991dd0999daa/Natural_PreRegisteredSufficiencySaturation/trace.json"
)
LLAMA_RESEARCH_COMPLETE_TRACE = (
    config.DEFAULT_PRODUCTION_ROOT
    / "traces/llama8b/branches/hotpot/5ae347fb5542992e3233c368/Natural_epistemic_confirmation/trace.json"
)
QWEN_CLEAN_TRACE = (
    config.DEFAULT_PRODUCTION_ROOT
    / "traces/qwen14b/branches/hotpot/5a7782b95542995d8318117f/NaturalClean/trace.json"
)

_FIXTURES_AVAILABLE = QWEN_NO_TOOL_EXIT_TRACE.exists() and LLAMA_RESEARCH_COMPLETE_TRACE.exists()


@unittest.skipUnless(_FIXTURES_AVAILABLE, "requires production trace fixtures on this machine")
class TraceParserFixtureTests(unittest.TestCase):
    def test_no_tool_exit_at_post_branch_step_1(self):
        """Hand-verified: decision 0 issues the first search+think_tool; decision 1
        (the first post-branch decision) has empty tool_calls -> no-tool exit at T_i=1."""
        pt = parse_trace_file(Path(QWEN_NO_TOOL_EXIT_TRACE))
        ep0 = pt.first_episode
        self.assertTrue(ep0.branch_reached)
        self.assertEqual(ep0.terminal_post_branch_index, 1)
        self.assertEqual(ep0.terminal_route, config.ROUTE_NO_TOOL_EXIT)

    def test_research_complete_route(self):
        """Hand-verified: decisions are [tavily_search, think_tool, think_tool,
        ResearchComplete] -> post-branch decisions are [think_tool, think_tool,
        ResearchComplete], terminal at post-branch step 3."""
        pt = parse_trace_file(Path(LLAMA_RESEARCH_COMPLETE_TRACE))
        ep0 = pt.first_episode
        self.assertEqual(ep0.terminal_post_branch_index, 3)
        self.assertEqual(ep0.terminal_route, config.ROUTE_RESEARCH_COMPLETE)
        self.assertEqual(pt.model, "llama8b")

    def test_multiple_researcher_episodes_supervisor_redelegation(self):
        """Hand-verified: 3 researcher subgraph episodes (2 supervisor redelegations),
        first episode terminates via no-tool-exit at post-branch step 3."""
        pt = parse_trace_file(Path(QWEN_REDELEGATION_TRACE))
        self.assertEqual(len(pt.episodes), 3)
        self.assertEqual(pt.supervisor_redelegation_count, 2)
        ep0 = pt.first_episode
        self.assertEqual(ep0.terminal_post_branch_index, 3)
        self.assertEqual(ep0.terminal_route, config.ROUTE_NO_TOOL_EXIT)
        # Redelegation must not change the FIRST attacked episode's own PCR outcome.
        self.assertEqual(ep0.episode_order, 0)

    def test_clean_condition_reaches_a_terminal_route(self):
        pt = parse_trace_file(Path(QWEN_CLEAN_TRACE))
        self.assertEqual(pt.condition, "NaturalClean")
        ep0 = pt.first_episode
        self.assertTrue(ep0.branch_reached)
        self.assertIsNotNone(ep0.terminal_route)

    def test_researcher_steps_are_far_fewer_than_raw_langgraph_events(self):
        """Confirms PCR step-counting uses researcher decisions, not raw event
        indices -- a trace can have thousands of raw events (mostly
        token-level streaming chunks) but only a handful of researcher
        decisions."""
        pt = parse_trace_file(Path(QWEN_NO_TOOL_EXIT_TRACE))
        import json

        raw = json.loads(Path(QWEN_NO_TOOL_EXIT_TRACE).read_text())
        n_raw_events = len(raw["events"])
        n_decisions = sum(len(ep.decisions) for ep in pt.episodes)
        self.assertGreater(n_raw_events, 100)
        self.assertLess(n_decisions, 20)
        self.assertLess(n_decisions, n_raw_events)

    def test_deterministic_repeated_parse(self):
        """Parsing the same trace twice must yield identical PCR-relevant outcomes."""
        a = parse_trace_file(Path(QWEN_REDELEGATION_TRACE))
        b = parse_trace_file(Path(QWEN_REDELEGATION_TRACE))
        self.assertEqual(a.first_episode.terminal_post_branch_index, b.first_episode.terminal_post_branch_index)
        self.assertEqual(a.first_episode.terminal_route, b.first_episode.terminal_route)
        self.assertEqual(len(a.episodes), len(b.episodes))


if __name__ == "__main__":
    unittest.main()

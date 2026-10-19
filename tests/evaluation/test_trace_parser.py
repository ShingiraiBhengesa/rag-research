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

from prematurestop.evaluation import config
from prematurestop.evaluation.trace_parser import _parse_decision, _reconstruct_imitated_tool_call, parse_trace_file

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


GEMMA_IMITATED_TOOL_CALL_TRACE = (
    config.DEFAULT_PRODUCTION_ROOT
    / "traces/gemma27b/branches/hotpot/5ae785685542994a481bbd47/Natural_PreRegisteredSufficiencySaturation/trace.json"
)


class ImitatedToolCallReconstructionTests(unittest.TestCase):
    """Regression tests for the 2026-09-28 fix: gemma27b sometimes imitates
    deep_researcher.py::_plain_turn_from_message's own gemma-history-rendering
    text ("Assistant requested tool call:\\n{json}") instead of emitting a
    real structured tool call, causing it to misclassify as NO_TOOL_DIRECT_EXIT
    (found: 93% of a 60-row sample of gemma27b's "NO_TOOL_DIRECT_EXIT" rows
    were this pattern, not a genuine no-further-action decision).
    """

    def test_reconstructs_a_well_formed_tavily_search_attempt(self):
        content = (
            "Okay, excellent! The first source directly answers the question. "
            "I will perform another search to confirm.\n\n"
            "Assistant requested tool call:\n"
            '{"args": {"queries": ["1981 World Rowing Championships location"]}, '
            '"id": "chatcmpl-tool-91a9149999999999", "name": "tavily_search"}'
        )
        result = _reconstruct_imitated_tool_call(content)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["name"], "tavily_search")
        self.assertEqual(result[0]["args"], {"queries": ["1981 World Rowing Championships location"]})
        self.assertEqual(result[0]["id"], "chatcmpl-tool-91a9149999999999")

    def test_reconstructs_a_think_tool_attempt_with_empty_args(self):
        content = (
            "Now I need to use think_tool to assess if I have enough information.\n\n"
            'Assistant requested tool call:\n{"args": {}, "id": "chatcmpl-tool-9114a7499499911a", "name": "think_tool"}'
        )
        result = _reconstruct_imitated_tool_call(content)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["name"], "think_tool")
        self.assertEqual(result[0]["args"], {})

    def test_does_not_match_a_genuine_final_answer(self):
        content = (
            "I believe I have confidently answered the question. No further searching is needed.\n\n"
            "Assistant response:\nThe songwriter is John D. Loudermilk."
        )
        self.assertIsNone(_reconstruct_imitated_tool_call(content))

    def test_does_not_match_plain_content_with_no_marker(self):
        content = "This is just ordinary reasoning text with no tool call markers at all."
        self.assertIsNone(_reconstruct_imitated_tool_call(content))

    def test_ignores_malformed_json_after_the_marker(self):
        content = "Assistant requested tool call:\nnot actually valid json {{{"
        self.assertIsNone(_reconstruct_imitated_tool_call(content))

    def test_ignores_a_payload_missing_a_name(self):
        content = 'Assistant requested tool call:\n{"args": {"queries": ["x"]}, "id": "abc"}'
        self.assertIsNone(_reconstruct_imitated_tool_call(content))

    def test_parse_decision_end_to_end_reconstructs_and_flags_it(self):
        """A real, empty-tool_calls researcher decision whose content imitates
        the marker text must still classify as has_tool_calls=False / a
        no-tool exit (the imitated call was never actually executed by the
        live graph -- see the module-level comment in trace_parser.py),
        while reconstructed_from_imitated_history_text=True marks it as an
        imitation artifact so canonical_completion can route it to
        TOOL_OR_PARSER_FAILURE instead of a genuine NO_TOOL_DIRECT_EXIT."""
        chain_end_event = {
            "data": {
                "output": (
                    "Command(update={'researcher_messages': [AIMessage("
                    'content=\'Okay, excellent! Confirmed.\\n\\nAssistant requested tool call:\\n'
                    '{"args": {"queries": ["x"]}, "id": "chatcmpl-tool-abc", "name": "tavily_search"}\', '
                    "additional_kwargs={}, response_metadata={}, id='lc_run--x', "
                    "tool_calls=[], invalid_tool_calls=[])]}, goto='researcher_tools')"
                )
            }
        }
        result = _parse_decision(1, "ep0", chain_end_event)
        self.assertTrue(result.parse_ok)
        self.assertFalse(result.has_tool_calls)
        self.assertEqual(result.tool_call_names, [])
        self.assertTrue(result.is_no_tool_exit)
        self.assertTrue(result.reconstructed_from_imitated_history_text)

    def test_parse_decision_leaves_a_real_tool_call_unreconstructed(self):
        """A real, non-empty tool_calls= list must never be routed through
        the reconstruction path (reconstructed_from_imitated_history_text
        stays False) -- confirms zero behavior change for qwen/llama/gpt-oss
        and gemma27b's own well-formed decisions."""
        chain_end_event = {
            "data": {
                "output": (
                    "Command(update={'researcher_messages': [AIMessage("
                    "content='', additional_kwargs={}, response_metadata={}, id='lc_run--y', "
                    "tool_calls=[{'name': 'tavily_search', 'args': {'queries': ['x']}, "
                    "'id': 'call_1', 'type': 'tool_call'}], invalid_tool_calls=[])]}, goto='researcher_tools')"
                )
            }
        }
        result = _parse_decision(0, "ep0", chain_end_event)
        self.assertTrue(result.has_tool_calls)
        self.assertFalse(result.reconstructed_from_imitated_history_text)

    # The fixture is a live production trace, not a frozen copy. This example was
    # hand-verified on 2026-10-07; the Gemma shard holding it was re-run on
    # 2026-10-11 and the new trace no longer contains imitated tool-call text.
    # So the test runs only while the fixture still shows the pattern it checks.
    # The parsing logic itself is covered by the synthetic tests above.
    @unittest.skipUnless(
        GEMMA_IMITATED_TOOL_CALL_TRACE.exists()
        and "Assistant requested tool call" in GEMMA_IMITATED_TOOL_CALL_TRACE.read_text(encoding="utf-8"),
        "production fixture missing, or re-run since it was hand-verified (no imitated tool-call text)",
    )
    def test_real_gemma27b_trace_second_decision_is_reconstructed(self):
        """Hand-verified real trace: decision 0 is a genuine tavily_search
        tool call; decision 1 has empty tool_calls= but its content imitates
        the marker text for a second tavily_search attempt."""
        import json

        trace = json.loads(GEMMA_IMITATED_TOOL_CALL_TRACE.read_text())
        decisions = [
            e for e in trace.get("raw_odr_events", [])
            if e.get("name") == "researcher" and e.get("event") == "on_chain_end"
        ]
        self.assertEqual(len(decisions), 2)

        first = _parse_decision(decisions[0].get("step", 0), None, decisions[0])
        self.assertTrue(first.has_tool_calls)
        self.assertFalse(first.reconstructed_from_imitated_history_text)

        second = _parse_decision(decisions[1].get("step", 1), None, decisions[1])
        self.assertFalse(second.has_tool_calls)
        self.assertEqual(second.tool_call_names, [])
        self.assertTrue(second.is_no_tool_exit)
        self.assertTrue(second.reconstructed_from_imitated_history_text)


if __name__ == "__main__":
    unittest.main()

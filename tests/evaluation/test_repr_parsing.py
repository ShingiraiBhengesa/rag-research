"""Unit tests for repr_parsing.py using synthetic (not real-data-dependent) strings
that mirror the exact shapes observed in real Qwen/Llama traces."""

from __future__ import annotations

import unittest

from prematurestop.evaluation.repr_parsing import LiteralScanError, extract_last_kwarg


class ReprParsingTests(unittest.TestCase):
    def test_tool_calls_not_confused_with_invalid_tool_calls(self):
        text = (
            "Command(update={'researcher_messages': [AIMessage(content='', "
            "tool_calls=[{'name': 'tavily_search', 'args': {'queries': ['a', 'b']}, "
            "'id': 'x', 'type': 'tool_call'}], invalid_tool_calls=[])]}, goto='researcher_tools')"
        )
        result = extract_last_kwarg(text, "tool_calls")
        self.assertIsNotNone(result)
        self.assertEqual(len(result.value), 1)
        self.assertEqual(result.value[0]["name"], "tavily_search")

    def test_empty_tool_calls_parses_as_empty_list_not_none(self):
        text = "AIMessage(content='done', tool_calls=[], invalid_tool_calls=[])"
        result = extract_last_kwarg(text, "tool_calls")
        self.assertEqual(result.value, [])

    def test_research_complete_tool_call_detected(self):
        text = (
            "AIMessage(content='', tool_calls=[{'name': 'ResearchComplete', 'args': {}, "
            "'id': 'y', 'type': 'tool_call'}], invalid_tool_calls=[])"
        )
        result = extract_last_kwarg(text, "tool_calls")
        names = [tc["name"] for tc in result.value]
        self.assertIn("ResearchComplete", names)

    def test_nested_quotes_inside_string_values_do_not_break_bracket_matching(self):
        text = (
            "AIMessage(content=\"it's a test with ] and [ brackets\", "
            "tool_calls=[{'name': 'think_tool', 'args': {'reflection': \"contains ']' char\"}, "
            "'id': 'z', 'type': 'tool_call'}], invalid_tool_calls=[])"
        )
        result = extract_last_kwarg(text, "tool_calls")
        self.assertEqual(len(result.value), 1)
        self.assertEqual(result.value[0]["name"], "think_tool")

    def test_missing_key_returns_none(self):
        result = extract_last_kwarg("AIMessage(content='no tool calls key here')", "tool_calls")
        self.assertIsNone(result)

    def test_llama_stringified_list_args_still_parses_as_a_string(self):
        text = (
            "AIMessage(content='', tool_calls=[{'name': 'tavily_search', "
            "'args': {'queries': \"['a', 'b']\"}, 'id': 'w', 'type': 'tool_call'}], "
            "invalid_tool_calls=[])"
        )
        result = extract_last_kwarg(text, "tool_calls")
        self.assertIsInstance(result.value[0]["args"]["queries"], str)


if __name__ == "__main__":
    unittest.main()

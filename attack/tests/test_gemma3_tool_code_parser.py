import json
import unittest

try:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from attack.runtime.gemma3_tool_code_parser import Gemma3ToolCodeParser

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


@unittest.skipUnless(
    _VLLM_AVAILABLE,
    "vllm is only installed in the 'vllm' conda environment, not 'whitebox' -- "
    "run this file directly under /home/sbhengesa/.conda/envs/vllm/bin/python "
    "to actually exercise it.",
)
class TestGemma3ToolCodeParser(unittest.TestCase):
    """Regression tests against real gemma27b production trace text
    (results/traces/gemma27b/, sampled 2026-09-14), not synthetic examples --
    these are the actual two sub-formats Gemma-3-27B-IT was observed to emit.
    """

    def setUp(self):
        self.parser = Gemma3ToolCodeParser(tokenizer=None, tools=None)
        self.request = ChatCompletionRequest(model="google/gemma-3-27b-it", messages=[])

    def test_1_python_call_style_tavily_search_normalizes_query_to_queries_list(self):
        text = (
            "Let's begin with the first search.\n\n"
            '```tool_code\ntavily_search(query="longest reigning British monarch spouse")\n```'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call.function.name, "tavily_search")
        args = json.loads(call.function.arguments)
        self.assertEqual(args, {"queries": ["longest reigning British monarch spouse"]})
        self.assertIn("Let's begin", result.content)

    def test_2_python_call_style_think_tool_normalizes_text_to_reflection(self):
        text = (
            "let's use `think_tool` to assess the results and plan the next step.\n\n"
            '```tool_code\nthink_tool(text="The initial search successfully identified '
            'Queen Elizabeth II as the longest-reigning monarch.")\n```'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        call = result.tool_calls[0]
        self.assertEqual(call.function.name, "think_tool")
        args = json.loads(call.function.arguments)
        self.assertEqual(
            args,
            {"reflection": "The initial search successfully identified Queen Elizabeth II as the longest-reigning monarch."},
        )

    def test_3_json_object_style_think_tool_maps_input_to_reflection(self):
        text = (
            "Now, let's start with the first research call.\n\n"
            '```tool_code\n{\n  "tool": "think_tool",\n  "input": '
            '"My plan is to use a single agent to find the requested information."\n}\n```'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        call = result.tool_calls[0]
        self.assertEqual(call.function.name, "think_tool")
        args = json.loads(call.function.arguments)
        self.assertEqual(
            args,
            {"reflection": "My plan is to use a single agent to find the requested information."},
        )

    def test_4_json_object_style_search_maps_input_to_queries_list(self):
        text = '```tool_code\n{"tool": "tavily_search", "input": "Prince Philip birthdate"}\n```'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        call = result.tool_calls[0]
        self.assertEqual(call.function.name, "tavily_search")
        args = json.loads(call.function.arguments)
        self.assertEqual(args, {"queries": ["Prince Philip birthdate"]})

    def test_5_no_tool_code_block_returns_content_only(self):
        text = "This is just a plain research summary with no tool call in it at all."
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_6_malformed_python_call_syntax_error_falls_back_to_no_tool_call(self):
        text = '```tool_code\ntavily_search(query=\n```'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)

    def test_7_real_queries_list_argument_passes_through_unwrapped(self):
        text = '```tool_code\ntavily_search(queries=["a", "b"])\n```'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        args = json.loads(result.tool_calls[0].function.arguments)
        self.assertEqual(args, {"queries": ["a", "b"]})

    def test_8_unrelated_tool_name_passes_arguments_through_unmodified(self):
        text = '```tool_code\nResearchComplete()\n```'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "ResearchComplete")
        self.assertEqual(json.loads(result.tool_calls[0].function.arguments), {})

    def _stream(self, chunks):
        """Feed text chunks through extract_tool_calls_streaming one at a
        time, as vLLM would for a real streamed generation, and collect
        every non-None DeltaMessage returned."""
        previous_text = ""
        deltas = []
        for chunk in chunks:
            current_text = previous_text + chunk
            result = self.parser.extract_tool_calls_streaming(
                previous_text, current_text, chunk, [], [], [], self.request
            )
            if result is not None:
                deltas.append(result)
            previous_text = current_text
        return deltas

    def test_9_streaming_real_forced_think_tool_json_shape_eventually_emits_tool_call(self):
        # This is the exact shape found in a real gemma27b gate trace
        # (2026-09-14): {"args": {...}, "id": "...", "name": "think_tool"} --
        # not {"tool": ..., "input": ...} like the earlier synthetic samples.
        full_text = (
            "Okay, let's think about this.\n\n"
            '```tool_code\n{"args": {"research_topic": "test topic"}, '
            '"id": "forced_think_tool_1", "name": "think_tool"}\n```'
        )
        # Simulate vLLM's real per-token streaming granularity: several
        # small chunks, not one chunk per line.
        chunks = [full_text[i : i + 7] for i in range(0, len(full_text), 7)]
        deltas = self._stream(chunks)

        tool_call_deltas = [d for d in deltas if d.tool_calls]
        self.assertEqual(len(tool_call_deltas), 1)
        function = tool_call_deltas[0].tool_calls[0].function
        self.assertEqual(function.name, "think_tool")
        args = json.loads(function.arguments)
        self.assertEqual(args, {"reflection": "test topic"})

    def test_10_streaming_content_before_block_is_emitted_and_partial_block_is_withheld(self):
        full_text = 'Let me search.\n\n```tool_code\ntavily_search(query="x")\n```'
        chunks = [full_text[i : i + 5] for i in range(0, len(full_text), 5)]
        deltas = self._stream(chunks)

        content_so_far = "".join(d.content for d in deltas if d.content)
        self.assertEqual(content_so_far, "Let me search.\n\n")

        tool_call_deltas = [d for d in deltas if d.tool_calls]
        self.assertEqual(len(tool_call_deltas), 1)
        self.assertEqual(tool_call_deltas[0].tool_calls[0].function.name, "tavily_search")

    def test_11_streaming_no_tool_call_passes_all_content_through(self):
        full_text = "Just a plain final answer with no tool call at all."
        chunks = [full_text[i : i + 6] for i in range(0, len(full_text), 6)]
        deltas = self._stream(chunks)
        content_so_far = "".join(d.content for d in deltas if d.content)
        self.assertEqual(content_so_far, full_text)
        self.assertFalse(any(d.tool_calls for d in deltas))


if __name__ == "__main__":
    unittest.main()

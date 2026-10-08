import unittest

try:
    from openai_harmony import HarmonyError

    from attack.runtime.gptoss_lenient_harmony_parser import (
        _CrashTolerantHarmonyParser,
        _lenient_get_streamable_parser_for_assistant,
        apply_patch,
    )
    from vllm.entrypoints.openai.parser.harmony_utils import (
        get_encoding,
        parse_output_into_messages,
    )
    from vllm.tool_parsers.openai_tool_parser import OpenAIToolParser

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


@unittest.skipUnless(
    _VLLM_AVAILABLE,
    "vllm/openai_harmony are only installed in the 'vllm' conda environment, "
    "not 'whitebox' -- run this file directly under "
    "/home/sbhengesa/.conda/envs/vllm/bin/python to actually exercise it.",
)
class TestGptOssLenientHarmonyParser(unittest.TestCase):
    """Regression tests against real gpt-oss-20b production error text
    (results/runs/gpt_oss_20b/, sampled 2026-09-20 across 627 row-level
    failures) -- these are the actual malformed harmony header patterns
    observed, not synthetic examples. No model or GPU is used anywhere in
    this file: every input is hand-constructed harmony-formatted text,
    encoded to token IDs via openai_harmony's own tokenizer/encoding, and
    fed directly into vLLM's real (installed) parsing code paths.
    """

    @classmethod
    def setUpClass(cls):
        apply_patch()
        cls.enc = get_encoding()

    class _FakeRequest:
        pass

    # -- Malformed inputs (each string is a real observed error pattern,
    # or a minimal reproduction of its class) -----------------------------

    MALFORMED_SAMPLES = {
        "duplicated_recipient_conduct_research": (
            '<|channel|>commentary to=functions.ConductResearch '
            'to=functions.ConductResearch<|constrain|>json<|message|>'
            '{"research_topic": "test"}<|call|>'
        ),
        "duplicated_recipient_tavily_search": (
            '<|channel|>commentary to=functions.tavily_search '
            'to=functions.tavily_search<|constrain|>json<|message|>'
            '{"queries": ["x"]}<|call|>'
        ),
        "duplicated_recipient_think_tool": (
            '<|channel|>commentary to=functions.think_tool '
            'to=functions.think_tool<|constrain|>json<|message|>'
            '{"reflection": "y"}<|call|>'
        ),
        "duplicated_recipient_research_complete": (
            '<|channel|>commentary to=functions.ResearchComplete '
            'to=functions.ResearchComplete<|message|>{}<|call|>'
        ),
        "trailing_garbage_recipient_after_call": (
            '<|channel|>commentary to=functions.tavily_search<|constrain|>json'
            '<|message|>{"queries": ["x"]}<|call|> to=functions.tavily_search'
        ),
    }

    def test_non_streaming_malformed_inputs_do_not_raise(self):
        for name, text in self.MALFORMED_SAMPLES.items():
            with self.subTest(sample=name):
                token_ids = self.enc.encode(text, allowed_special="all")
                parser = parse_output_into_messages(token_ids)  # must not raise
                self.assertTrue(getattr(parser, "crashed", False))
                self.assertIsInstance(parser.crash_error, HarmonyError)

    def test_tool_parser_extract_tool_calls_malformed_inputs_do_not_raise(self):
        tp = OpenAIToolParser(tokenizer=None, tools=None)
        for name, text in self.MALFORMED_SAMPLES.items():
            with self.subTest(sample=name):
                token_ids = self.enc.encode(text, allowed_special="all")
                result = tp.extract_tool_calls(
                    model_output="", request=self._FakeRequest(), token_ids=token_ids
                )  # must not raise
                # Graceful degradation: at minimum no unhandled exception;
                # a message fully parsed before the malformed continuation
                # (see trailing_garbage sample) may still be recovered.
                self.assertIsInstance(result.tools_called, bool)

    def test_trailing_garbage_still_recovers_the_valid_leading_tool_call(self):
        text = self.MALFORMED_SAMPLES["trailing_garbage_recipient_after_call"]
        tp = OpenAIToolParser(tokenizer=None, tools=None)
        token_ids = self.enc.encode(text, allowed_special="all")
        result = tp.extract_tool_calls(
            model_output="", request=self._FakeRequest(), token_ids=token_ids
        )
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "tavily_search")

    def test_streaming_malformed_inputs_do_not_raise(self):
        for name, text in self.MALFORMED_SAMPLES.items():
            with self.subTest(sample=name):
                parser = _lenient_get_streamable_parser_for_assistant()
                token_ids = self.enc.encode(text, allowed_special="all")
                for token_id in token_ids:
                    parser.process(token_id)  # must not raise
                self.assertTrue(parser.crashed)

    # -- Well-formed inputs: confirm zero regression ------------------------

    def test_well_formed_tool_call_is_extracted_unchanged(self):
        text = (
            '<|channel|>commentary to=functions.tavily_search<|constrain|>json'
            '<|message|>{"queries": ["squadron no 5 release date"]}<|call|>'
        )
        tp = OpenAIToolParser(tokenizer=None, tools=None)
        token_ids = self.enc.encode(text, allowed_special="all")
        parser = parse_output_into_messages(token_ids)
        result = tp.extract_tool_calls(
            model_output="", request=self._FakeRequest(), token_ids=token_ids
        )

        self.assertFalse(getattr(parser, "crashed", False))
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "tavily_search")
        self.assertEqual(result.tool_calls[0].function.arguments, '{"queries": ["squadron no 5 release date"]}')
        self.assertIsNone(result.content)

    def test_well_formed_think_tool_call_is_extracted_unchanged(self):
        text = (
            '<|channel|>commentary to=functions.think_tool<|constrain|>json'
            '<|message|>{"reflection": "need more info"}<|call|>'
        )
        tp = OpenAIToolParser(tokenizer=None, tools=None)
        token_ids = self.enc.encode(text, allowed_special="all")
        result = tp.extract_tool_calls(
            model_output="", request=self._FakeRequest(), token_ids=token_ids
        )
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "think_tool")

    def test_well_formed_final_content_is_extracted_unchanged(self):
        text = "<|channel|>final<|message|>The answer is 42."
        tp = OpenAIToolParser(tokenizer=None, tools=None)
        token_ids = self.enc.encode(text, allowed_special="all")
        parser = parse_output_into_messages(token_ids)
        result = tp.extract_tool_calls(
            model_output="", request=self._FakeRequest(), token_ids=token_ids
        )

        self.assertFalse(getattr(parser, "crashed", False))
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, "The answer is 42.")

    def test_streaming_well_formed_final_content_yields_identical_delta_text(self):
        text = "<|channel|>final<|message|>The answer is 42."
        parser = _lenient_get_streamable_parser_for_assistant()
        token_ids = self.enc.encode(text, allowed_special="all")
        deltas = []
        for token_id in token_ids:
            parser.process(token_id)
            deltas.append(parser.last_content_delta or "")
        self.assertFalse(parser.crashed)
        self.assertEqual("".join(deltas), "The answer is 42.")

    # -- Post-crash state is safe to read, matching the wrapper's assumptions --

    def test_real_parser_is_safely_readable_after_a_caught_crash(self):
        from openai_harmony import Role, StreamableParser

        text = self.MALFORMED_SAMPLES["duplicated_recipient_conduct_research"]
        token_ids = self.enc.encode(text, allowed_special="all")
        raw_parser = StreamableParser(self.enc, role=Role.ASSISTANT)
        crashed = False
        for token_id in token_ids:
            try:
                raw_parser.process(token_id)
            except HarmonyError:
                crashed = True
                break
        self.assertTrue(crashed)
        # Reading state after the crash must not itself raise.
        self.assertIsNone(raw_parser.current_channel)
        self.assertEqual(raw_parser.messages, [])

    def test_patch_is_idempotent(self):
        # Calling apply_patch() twice (as both api_server.py call sites do)
        # must not double-wrap or error.
        apply_patch()
        apply_patch()
        parser = _lenient_get_streamable_parser_for_assistant()
        self.assertIsInstance(parser, _CrashTolerantHarmonyParser)


if __name__ == "__main__":
    unittest.main()

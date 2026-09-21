# SPDX-License-Identifier: Apache-2.0
"""Custom vLLM tool-call parser for google/gemma-3-27b-it.

Not part of vLLM itself -- loaded via ``--tool-parser-plugin`` at server
startup (see attack/runtime/runtime_gate.py / generate_production_arrays.py).

## Why this exists

vLLM ships two Gemma-named parsers, and neither matches this model:

* ``functiongemma`` is documented (see its own module docstring) as being
  for ``google/functiongemma-270m-it``, a small, separately fine-tuned model
  that emits ``<start_function_call>call:name{...}<end_function_call>``.
  ``google/gemma-3-27b-it`` never emits those tokens.
* ``gemma4`` is for a different, newer Gemma generation's
  ``<|tool_call>call:name{...}<tool_call|>`` format. Also not this model.

Neither vLLM feature request for native Gemma-3 tool calling
(vllm-project/vllm#15403, #16482) was implemented upstream ("closed as not
planned" as of 2026-09).

## What Gemma-3-27B-IT actually emits

Confirmed by direct inspection of real completed production traces
(``results/traces/gemma27b/``, 2026-09-14): Gemma-3-27B-IT writes its
intended tool call as a markdown code block fenced with the language tag
``tool_code``, in one of two inconsistent sub-formats:

1. A Python-call expression:  ```tool_code\\ntavily_search(query="...")\\n```
2. A JSON object:             ```tool_code\\n{"tool": "think_tool", "input": "..."}\\n```

Neither matches this project's actual tool schemas exactly, even once the
fence is stripped:

* ``tavily_search`` takes ``queries: list[str]`` (see
  ``attack/runtime/benchmark_search.py``); Gemma writes a singular
  ``query`` string (Python-call style) or a bare ``input`` string
  (JSON-object style).
* ``think_tool`` takes ``reflection: str`` (see
  ``open_deep_research/src/open_deep_research/utils.py``); Gemma writes
  ``text`` (Python-call style) or ``input`` (JSON-object style).

This parser recognizes both sub-formats and normalizes the argument name/
shape to what the real tool schema expects, rather than requiring an exact
literal match vLLM's built-in parsers don't attempt to provide either.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager

logger = init_logger(__name__)

# Argument-name/shape normalization for the two researcher-subgraph tools
# this pipeline actually exposes. Keyed by (tool_name, whatever key Gemma
# used) -> the real schema's argument name; "queries" values are wrapped in
# a list since the real schema is list[str], not str.
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "tavily_search": {"query": "queries", "queries": "queries", "input": "queries"},
    "think_tool": {"text": "reflection", "reflection": "reflection", "input": "reflection"},
}
_LIST_ARGS = {"queries"}
# Both real tools this pipeline exposes to the researcher subgraph take
# exactly one argument. Observed in real traces: Gemma sometimes names that
# single argument after a DIFFERENT tool's parameter entirely (e.g. calling
# think_tool with a "research_topic" key -- ConductResearch's parameter
# name, not think_tool's "reflection") rather than an otherwise-reasonable
# alias. When exactly one argument is given and its key matches neither an
# alias nor the tool's own real name, treat it positionally as that one
# argument rather than passing an unrecognized key through unchanged.
_PRIMARY_ARG = {"tavily_search": "queries", "think_tool": "reflection"}


def _normalize_arguments(func_name: str, raw_args: dict) -> dict:
    aliases = _ARG_ALIASES.get(func_name)
    if aliases is None:
        return raw_args
    normalized: dict = {}
    for key, value in raw_args.items():
        real_key = aliases.get(key, key)
        if real_key not in aliases.values() and len(raw_args) == 1 and func_name in _PRIMARY_ARG:
            real_key = _PRIMARY_ARG[func_name]
        if real_key in _LIST_ARGS and not isinstance(value, list):
            value = [value]
        normalized[real_key] = value
    return normalized


@ToolParserManager.register_module("gemma3_tool_code")
class Gemma3ToolCodeParser(ToolParser):
    """Tool parser for google/gemma-3-27b-it's ```tool_code``` markdown format.

    Loaded via:
        --tool-parser-plugin <absolute path to this file> \\
        --tool-call-parser gemma3_tool_code
    """

    tool_code_block_regex: re.Pattern = re.compile(
        r"```tool_code\s*\n(.*?)```", re.DOTALL
    )
    # A single top-level call: name(arg=val, ...) -- args parsed with
    # ast.literal_eval via a synthetic `f(...)` Call node, never eval().
    python_call_regex: re.Pattern = re.compile(
        r"^\s*(\w+)\s*\((.*)\)\s*$", re.DOTALL
    )

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # Streaming state: this pipeline runs ODR under LangGraph's
        # .astream_events(), which makes LangChain's ChatOpenAI integration
        # request token-level streaming from vLLM even though each node
        # just does a plain `await model.ainvoke(...)` -- confirmed
        # empirically (on_chat_model_stream fires ~2000x per generation in
        # real traces). So extract_tool_calls_streaming, not
        # extract_tool_calls, is what actually runs in production.
        self._tool_call_emitted = False

    @staticmethod
    def _parse_python_call(body: str) -> tuple[str, dict] | None:
        match = Gemma3ToolCodeParser.python_call_regex.match(body.strip())
        if not match:
            return None
        func_name, args_str = match.group(1), match.group(2)
        try:
            call_node = ast.parse(f"f({args_str})", mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(call_node, ast.Call):
            return None
        args: dict = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                continue
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                return None
        return func_name, args

    @staticmethod
    def _parse_json_object(body: str) -> tuple[str, dict] | None:
        try:
            obj = json.loads(body.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        func_name = obj.get("tool") or obj.get("name")
        if not isinstance(func_name, str):
            return None
        if "input" in obj and isinstance(obj.get("args"), dict):
            args = obj["args"]
        elif "input" in obj:
            args = {"input": obj["input"]}
        elif isinstance(obj.get("args"), dict):
            args = obj["args"]
        else:
            args = {k: v for k, v in obj.items() if k not in ("tool", "name")}
        return func_name, args

    def _extract_one(self, body: str) -> ToolCall | None:
        parsed = self._parse_python_call(body) or self._parse_json_object(body)
        if parsed is None:
            return None
        func_name, raw_args = parsed
        arguments = _normalize_arguments(func_name, raw_args)
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=func_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        blocks = self.tool_code_block_regex.findall(model_output)
        if not blocks:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = [
                call for call in (self._extract_one(block) for block in blocks) if call
            ]
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            first_block_start = model_output.find("```tool_code")
            content = (
                model_output[:first_block_start].strip()
                if first_block_start > 0
                else None
            )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error extracting tool calls from Gemma-3 response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._tool_call_emitted:
            return None

        try:
            blocks = self.tool_code_block_regex.findall(current_text)
            if blocks:
                # A full ```tool_code ... ``` block just completed. Parse it
                # and emit the whole tool call in one shot -- there is no
                # value in trying to stream a ```tool_code block
                # token-by-token, since it must be fully buffered and
                # regex-matched anyway before it can be parsed at all.
                call = self._extract_one(blocks[0])
                self._tool_call_emitted = True
                if call is None:
                    return None
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=call.function.name,
                                arguments=call.function.arguments,
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )

            # No complete block yet. Emit only the prefix of current_text
            # that is safely known NOT to be part of a ```tool_code fence --
            # withhold any tail that could still grow into that marker, so a
            # marker split across delta-chunk boundaries never partially
            # leaks into visible content (e.g. one chunk ending exactly
            # after "```tool_c").
            marker = "```tool_code"
            marker_pos = current_text.find(marker)
            if marker_pos != -1:
                # The marker has fully appeared somewhere in current_text
                # but its block hasn't closed yet (the findall above would
                # have caught that already) -- withhold everything from the
                # marker onward, no matter how much more has streamed in.
                safe_len = marker_pos
            else:
                # The marker hasn't fully appeared. Withhold only a tail
                # that could still grow into it (e.g. a chunk boundary that
                # landed exactly after "```tool_c").
                safe_len = len(current_text)
                for prefix_len in range(min(len(marker) - 1, len(current_text)), 0, -1):
                    if current_text.endswith(marker[:prefix_len]):
                        safe_len = len(current_text) - prefix_len
                        break
            # previous_text is vLLM's own full-decoded-so-far text (not
            # tracked emission state), so starting from len(previous_text)
            # each call naturally re-attempts anything withheld last time
            # without double-emitting anything already returned.
            new_safe_content = current_text[len(previous_text) : safe_len]
            if new_safe_content:
                return DeltaMessage(content=new_safe_content)
            return None
        except Exception:
            logger.exception("Error in Gemma-3 streaming tool call extraction")
            return None

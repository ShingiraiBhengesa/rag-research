# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch: make vLLM's gpt-oss harmony-format parsing crash-tolerant.

Not part of vLLM itself -- loaded via ``--tool-parser-plugin`` at server
startup (see src/prematurestop/pipelines/fixed_cue_matrix/runtime_gate.py / generate_production_arrays.py),
exactly like src/prematurestop/system/tool_parsers/gemma3_tool_code.py. Despite the
``--tool-parser-plugin`` name, vLLM's loader (``ToolParserManager.import_tool_parser``)
just does a bare dynamic import of this file's path -- any top-level code
runs, including the monkey-patch below, whether or not this file also
registers a ``ToolParser`` subclass (it doesn't need to; see "Why this
doesn't touch OpenAIToolParser" below).

## Why this exists

``openai/gpt-oss-20b`` production traffic (results/traces/gpt_oss_20b/,
2026-09-14 onward) showed 627+ row-level failures, all
``APIError('unexpected tokens remaining in message header: ...')`` --
e.g. ``Some("to=functions.ConductResearch")``, ``Some("<|channel|>commentary")``,
``Some("to=functions.tavily_search")``. This is not a client-side
tool-call-extraction mismatch (unlike gemma27b's ``functiongemma`` bug,
fixed by ``gemma3_tool_code_parser.py``) -- gpt-oss-20b sometimes emits a
malformed harmony header (a duplicated/garbled ``to=functions.X``
recipient token, or chain-of-thought text bleeding into a channel/
recipient header segment instead of stopping at ``<|message|>``), and
vLLM's own server-side harmony state machine (``openai_harmony.StreamableParser.process``,
a Rust extension) raises ``HarmonyError`` and crashes the *entire request*
-- both the streaming path (``vllm/entrypoints/openai/chat_completion/serving.py``,
which calls ``harmony_parser.process(token_id)`` per token with no
try/except) and the non-streaming path (``vllm/entrypoints/openai/parser/harmony_utils.py::parse_output_into_messages``,
used by vLLM's built-in ``OpenAIToolParser``).

Confirmed via ``StreamableParser(..., strict=False)`` that the ``strict``
constructor flag does **not** suppress this error -- it is a structural
parse failure in the header token grammar, not a leniency-tunable
validation check. There is also no CLI flag or config to change this.

## The fix

Both vLLM call sites obtain their parser from the same one-line factory,
``get_streamable_parser_for_assistant()`` in ``harmony_utils.py``. This
module replaces that factory (in every module that imported it) with one
that returns a ``_CrashTolerantHarmonyParser``: a thin wrapper that
forwards every attribute to a real ``StreamableParser``, but catches
``HarmonyError`` from ``.process()``/``.process_eos()`` and, from that
point on, silently stops forwarding further tokens instead of
propagating the exception.

Verified (see tests/system/test_gptoss_lenient_harmony_parser.py) that
after a caught crash, reading ``.current_channel``/``.current_content``/
``.messages``/``.last_content_delta`` on the underlying real parser is
safe and returns an empty/no-content state (the crash happens before any
``<|message|>`` content is parsed for that malformed header), and that
calling ``.process()`` again on the *same* real parser after a crash
raises a *different* error (state-machine desync) -- confirming tokens
must stop being forwarded, not merely retried.

## Why this doesn't touch OpenAIToolParser

``OpenAIToolParser.extract_tool_calls`` (vllm/tool_parsers/openai_tool_parser.py)
already handles a parser with zero messages exactly like any legitimately
tool-call-free generation: ``tools_called=False``, ``content=None``. With
this patch, a malformed generation degrades into that same, already-
correct "no tool call this turn" outcome -- which this pipeline already
has a modeled path for (PCR's ``NO_TOOL_DIRECT_EXIT`` route, or ODR's
``force_benchmark_search`` safety net for benchmark-local retrieval) --
instead of crashing the row/shard with an unrecoverable ``APIError``.

## Scope

This does not change ``--tool-call-parser`` (stays "openai") or
``--reasoning-parser`` (stays "openai_gptoss") for gpt_oss_20b. It only
prevents vLLM's own server-side harmony token processing from raising on
a malformed sequence -- content already parsed before the malformed
token is preserved; content that would have followed it is lost for that
one generation, matching the existing "no tool call" outcome class this
pipeline already scores correctly.
"""

from __future__ import annotations

import logging

from openai_harmony import HarmonyError, StreamableParser

logger = logging.getLogger(__name__)


class _CrashTolerantHarmonyParser:
    """Wraps a real StreamableParser; swallows HarmonyError from .process()/
    .process_eos() by freezing forward progress instead of propagating."""

    def __init__(self, *args, **kwargs) -> None:
        self._parser = StreamableParser(*args, **kwargs)
        self._crashed = False
        self._crash_error: HarmonyError | None = None

    @property
    def crashed(self) -> bool:
        return self._crashed

    @property
    def crash_error(self) -> HarmonyError | None:
        return self._crash_error

    def process(self, token: int) -> "_CrashTolerantHarmonyParser":
        if self._crashed:
            return self
        try:
            self._parser.process(token)
        except HarmonyError as exc:
            self._crashed = True
            self._crash_error = exc
            logger.warning(
                "gpt-oss harmony parser hit a malformed token sequence; "
                "treating the remainder of this generation as no-tool-call "
                "content instead of raising: %r",
                exc,
            )
        return self

    def process_eos(self) -> "_CrashTolerantHarmonyParser":
        if self._crashed:
            return self
        try:
            self._parser.process_eos()
        except HarmonyError as exc:
            self._crashed = True
            self._crash_error = exc
            logger.warning(
                "gpt-oss harmony parser hit a malformed token sequence at "
                "EOS; treating the remainder as no-tool-call content: %r",
                exc,
            )
        return self

    def __getattr__(self, name):
        # Delegates everything not explicitly overridden above:
        # current_channel, current_content, current_content_type,
        # current_recipient, current_role, last_content_delta, messages,
        # state, state_data, tokens.
        return getattr(self._parser, name)


def _lenient_get_streamable_parser_for_assistant():
    from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
    from openai_harmony import Role

    return _CrashTolerantHarmonyParser(get_encoding(), role=Role.ASSISTANT)


def _patch_module(module) -> None:
    if getattr(module, "_gptoss_lenient_harmony_patch_applied", False):
        return
    module.get_streamable_parser_for_assistant = _lenient_get_streamable_parser_for_assistant
    module._gptoss_lenient_harmony_patch_applied = True


def apply_patch() -> None:
    """Idempotent; safe to call multiple times or import multiple times."""
    import vllm.entrypoints.openai.parser.harmony_utils as harmony_utils

    _patch_module(harmony_utils)

    try:
        import vllm.entrypoints.openai.chat_completion.serving as chat_serving

        _patch_module(chat_serving)
    except ImportError:
        pass

    try:
        import vllm.entrypoints.openai.responses.context as responses_context

        _patch_module(responses_context)
    except ImportError:
        pass


apply_patch()

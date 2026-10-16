"""Literal-safe extraction of values from LangGraph's ``repr()``-serialized trace fields.

Raw production traces store `Command(...)`/`AIMessage(...)` node outputs as Python
`repr()` strings (not JSON), e.g.::

    "Command(update={'researcher_messages': [AIMessage(content='', ...,
    tool_calls=[{'name': 'tavily_search', 'args': {...}, 'id': '...',
    'type': 'tool_call'}], invalid_tool_calls=[])]}, goto='researcher_tools')"

This module extracts a single ``key=<python-literal>`` kwarg out of such a
string without executing arbitrary code: it locates the key with a
word-boundary-safe regex (so ``tool_calls=`` never matches inside
``invalid_tool_calls=``), scans forward respecting quoting/escaping/bracket
nesting to find where the literal value ends, and evaluates only that
slice with :func:`ast.literal_eval`. Nothing outside the located slice is
executed or evaluated.

Verified against both Qwen (`qwen14b`, Hermes tool-call format) and Llama
(`llama8b`, llama3_json tool-call format) traces: both models serialize
`tool_calls` through the same LangChain `AIMessage.__repr__`, so the same
scanner applies, though Llama's `args` values are sometimes themselves a
stringified Python list (e.g. ``'queries': "['a', 'b']"``) rather than a
native list -- callers that need query text should handle both cases (see
:func:`prematurestop.evaluation.trace_parser.tool_call_query_strings`).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


class LiteralScanError(ValueError):
    """Raised when a kwarg literal could not be safely isolated or parsed."""


def _scan_literal_end(s: str, start: int) -> int:
    """Return the index one past the end of the Python literal starting at ``s[start]``."""
    if start >= len(s):
        raise LiteralScanError(f"scan start {start} is past end of string (len={len(s)})")
    c = s[start]

    if c in ("'", '"'):
        quote = c
        i = start + 1
        while i < len(s):
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == quote:
                return i + 1
            i += 1
        raise LiteralScanError("unterminated string literal")

    if c in "[{(":
        closers = {"[": "]", "{": "}", "(": ")"}
        stack = [closers[c]]
        i = start + 1
        in_str: str | None = None
        while i < len(s):
            ch = s[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
            else:
                if ch in ("'", '"'):
                    in_str = ch
                elif ch in "[{(":
                    stack.append(closers[ch])
                elif ch in "]})":
                    if not stack or stack[-1] != ch:
                        raise LiteralScanError(f"bracket mismatch at index {i}")
                    stack.pop()
                    if not stack:
                        return i + 1
            i += 1
        raise LiteralScanError("unterminated bracket literal")

    # number / True / False / None / bare token (e.g. an un-repr'd identifier)
    m = re.match(r"[A-Za-z0-9_.\-+]+", s[start:])
    if not m or not m.group(0):
        raise LiteralScanError(f"cannot scan literal at index {start}: {s[start:start + 40]!r}")
    return start + m.end()


@dataclass
class ExtractedKwarg:
    key: str
    raw_text: str
    value: object
    span: tuple[int, int]


def extract_last_kwarg(text: str, key: str) -> ExtractedKwarg | None:
    """Find the *last* top-level occurrence of ``key=<literal>`` in ``text``.

    A negative lookbehind on the preceding character prevents ``tool_calls=``
    from matching inside ``invalid_tool_calls=``. Returns ``None`` if the key
    is not present. Raises :class:`LiteralScanError` if the key is present
    but its value cannot be safely scanned/parsed -- callers must treat this
    as a parse failure, not silently default to an empty value.
    """
    matches = list(re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(key)}=", text))
    if not matches:
        return None
    match = matches[-1]
    start = match.end()
    end = _scan_literal_end(text, start)
    raw_text = text[start:end]
    try:
        value = ast.literal_eval(raw_text)
    except Exception as exc:  # noqa: BLE001 -- surfaced to caller as a parse failure
        raise LiteralScanError(f"ast.literal_eval failed for key={key!r}: {exc}") from exc
    return ExtractedKwarg(key=key, raw_text=raw_text, value=value, span=(start, end))


def extract_all_kwarg_occurrences(text: str, key: str) -> list[ExtractedKwarg]:
    """Like :func:`extract_last_kwarg` but returns every occurrence, in order."""
    out: list[ExtractedKwarg] = []
    for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(key)}=", text):
        start = match.end()
        try:
            end = _scan_literal_end(text, start)
            raw_text = text[start:end]
            value = ast.literal_eval(raw_text)
        except LiteralScanError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LiteralScanError(f"ast.literal_eval failed for key={key!r}: {exc}") from exc
        out.append(ExtractedKwarg(key=key, raw_text=raw_text, value=value, span=(start, end)))
    return out

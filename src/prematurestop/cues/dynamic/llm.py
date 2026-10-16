"""Minimal OpenAI-compatible JSON chat client for the attacker endpoint.

Structured output uses the standard `response_format: json_schema` field
(supported by vLLM >= 0.6; the serving env is vLLM 0.23). Every call is a
fresh stateless conversation with an explicit seed. The transport is
injectable so the whole cue pipeline is testable offline.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


class ChatClient(Protocol):
    def complete_json(self, messages: list[dict], *, schema: Mapping[str, Any], schema_name: str,
                      temperature: float, top_p: float, max_tokens: int, seed: int) -> "ChatResult": ...


@dataclass
class ChatResult:
    raw: str
    parsed: Any                 # decoded JSON, or None
    error: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request: dict = field(default_factory=dict)


Transport = Callable[[str, dict, float], dict]


def _http_transport(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


@dataclass
class OpenAICompatClient:
    base_url: str
    model: str
    timeout: float = 180.0
    max_network_retries: int = 4
    transport: Transport = _http_transport

    def complete_json(self, messages, *, schema, schema_name, temperature, top_p, max_tokens, seed) -> ChatResult:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": dict(schema)}},
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        last: Exception | None = None
        for attempt in range(self.max_network_retries):
            try:
                resp = self.transport(url, body, self.timeout)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(min(2 ** attempt, 15))
        else:
            raise RuntimeError(f"attacker endpoint failed after {self.max_network_retries} tries: {last!r}")
        usage = resp.get("usage") or {}
        raw = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        parsed, err = decode_json(raw)
        return ChatResult(raw=raw, parsed=parsed, error=err, prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"),
                          request={k: body[k] for k in ("temperature", "top_p", "max_tokens", "seed")})


def decode_json(raw: str) -> tuple[Any, str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.startswith("json") else text
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc}"

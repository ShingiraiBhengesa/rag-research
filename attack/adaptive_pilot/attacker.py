"""Attacker-model calls: one generation prompt and one refinement prompt.

Each call is a fresh, stateless two-message conversation, so no attacker state
carries across questions, conditions, or victim models. The victim model name
is never placed in the prompt (pilot v1). Sampling seeds are derived from the
question/condition/observation, not from the victim.

This module only renders prompts, calls the OpenAI-compatible vLLM endpoint,
and resamples when deterministic validation fails. It never selects a
candidate based on victim outcomes; refinement is driven solely by the
whitelisted feedback payload passed in by the runner.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from attack.adaptive_pilot.cue_specs import CueSpec
from attack.adaptive_pilot.feedback import assert_feedback_permitted
from attack.adaptive_pilot.validation import (
    REQUIRED_OUTPUT_KEYS,
    TokenCounter,
    ValidationResult,
    parse_attacker_output,
    validate_suffix,
)
from attack.runtime.hashing import hash_text, stable_hash

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
GENERATION_PROMPT_VERSION = "generation_v1"
REFINEMENT_PROMPT_VERSION = "refinement_v1"

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in REQUIRED_OUTPUT_KEYS},
    "required": list(REQUIRED_OUTPUT_KEYS),
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AttackerConfig:
    base_url: str
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    revision: str = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 400
    guided_json: bool = True
    max_validation_samples: int = 3  # constraint-compliance resamples, never outcome-based
    max_network_retries: int = 4
    request_timeout: float = 120.0

    def fingerprint(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "guided_json": self.guided_json,
            "max_validation_samples": self.max_validation_samples,
        }


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def prompt_version_hashes() -> dict:
    names = [
        "generation_v1.system.txt", "generation_v1.user.txt",
        "refinement_v1.system.txt", "refinement_v1.user.txt",
    ]
    return {name: hash_text(_read_prompt(name), length=None) for name in names}


def _bullet(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def _fill(template: str, mapping: Mapping[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"<<{key}>>", value)
    return out


def render_generation_messages(spec: CueSpec, question: str, observation: str) -> list[dict]:
    system = _fill(_read_prompt("generation_v1.system.txt"), {"TOKEN_LIMIT": str(spec.suffix_token_limit)})
    user = _fill(
        _read_prompt("generation_v1.user.txt"),
        {
            "CONDITION": spec.condition,
            "MECHANISM": spec.intended_mechanism,
            "REQUIRED": _bullet(spec.required_features),
            "PROHIBITED": _bullet(tuple(spec.prohibited_mechanisms) + tuple(spec.global_prohibitions)),
            "ORIGINAL": spec.original_suffix,
            "QUESTION": question,
            "OBSERVATION": observation,
        },
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_refinement_messages(
    spec: CueSpec, question: str, observation: str, previous_suffix: str, feedback_payload: Mapping[str, Any]
) -> list[dict]:
    assert_feedback_permitted(feedback_payload)
    system = _fill(_read_prompt("refinement_v1.system.txt"), {"TOKEN_LIMIT": str(spec.suffix_token_limit)})
    user = _fill(
        _read_prompt("refinement_v1.user.txt"),
        {
            "CONDITION": spec.condition,
            "MECHANISM": spec.intended_mechanism,
            "REQUIRED": _bullet(spec.required_features),
            "PROHIBITED": _bullet(tuple(spec.prohibited_mechanisms) + tuple(spec.global_prohibitions)),
            "ORIGINAL": spec.original_suffix,
            "QUESTION": question,
            "OBSERVATION": observation,
            "PREVIOUS_SUFFIX": previous_suffix,
            "FEEDBACK_JSON": json.dumps(dict(feedback_payload), sort_keys=True),
        },
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class AttackerCall:
    stage: str                       # "generation" | "refinement"
    prompt_version: str
    accepted: bool
    suffix: str
    mechanism_rationale: str
    samples: list[dict]              # one entry per resample attempt (raw + validation)
    n_samples: int
    seed: int
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float
    error: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _derive_seed(stage: str, condition: str, question: str, observation: str, attempt: int) -> int:
    digest = stable_hash(
        {"stage": stage, "condition": condition, "question": question, "observation_hash": hash_text(observation, length=None), "attempt": attempt},
        length=8,
    )
    return int(digest, 16) % (2**31 - 1)


def _post_chat(cfg: AttackerConfig, messages: list[dict], seed: int) -> dict:
    body: dict[str, Any] = {
        "model": cfg.model_id,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
        "seed": seed,
    }
    if cfg.guided_json:
        body["extra_body"] = {"guided_json": OUTPUT_JSON_SCHEMA}
        body["response_format"] = {"type": "json_object"}
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode()
    last_exc: Exception | None = None
    for attempt in range(cfg.max_network_retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=cfg.request_timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"attacker endpoint failed after {cfg.max_network_retries} tries: {last_exc!r}")


def _run_stage(
    cfg: AttackerConfig,
    stage: str,
    messages: list[dict],
    spec: CueSpec,
    question: str,
    observation: str,
    token_counter: TokenCounter | None,
    prompt_version: str,
) -> AttackerCall:
    started = time.perf_counter()
    samples: list[dict] = []
    accepted_suffix = ""
    accepted_rationale = ""
    accepted = False
    prompt_tokens = completion_tokens = None
    seed = _derive_seed(stage, spec.condition, question, observation, 0)
    for attempt in range(cfg.max_validation_samples):
        seed = _derive_seed(stage, spec.condition, question, observation, attempt)
        try:
            response = _post_chat(cfg, messages, seed)
        except RuntimeError as exc:
            samples.append({"attempt": attempt, "seed": seed, "error": repr(exc)})
            continue
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        raw = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed, parse_err = parse_attacker_output(raw)
        if parsed is None:
            samples.append({"attempt": attempt, "seed": seed, "raw": raw, "parse_error": parse_err})
            continue
        result: ValidationResult = validate_suffix(
            parsed["suffix"], spec=spec, question=question, clean_observation=observation, token_counter=token_counter
        )
        samples.append({"attempt": attempt, "seed": seed, "raw": raw, "validation": result.to_json()})
        if result.ok:
            accepted = True
            accepted_suffix = parsed["suffix"]
            accepted_rationale = parsed["mechanism_rationale"]
            break
    return AttackerCall(
        stage=stage,
        prompt_version=prompt_version,
        accepted=accepted,
        suffix=accepted_suffix,
        mechanism_rationale=accepted_rationale,
        samples=samples,
        n_samples=len(samples),
        seed=seed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=time.perf_counter() - started,
        error="" if accepted else "no_valid_candidate_within_sample_budget",
    )


def generate_suffix(
    cfg: AttackerConfig, spec: CueSpec, question: str, observation: str, token_counter: TokenCounter | None
) -> AttackerCall:
    messages = render_generation_messages(spec, question, observation)
    return _run_stage(cfg, "generation", messages, spec, question, observation, token_counter, GENERATION_PROMPT_VERSION)


def refine_suffix(
    cfg: AttackerConfig,
    spec: CueSpec,
    question: str,
    observation: str,
    previous_suffix: str,
    feedback_payload: Mapping[str, Any],
    token_counter: TokenCounter | None,
) -> AttackerCall:
    messages = render_refinement_messages(spec, question, observation, previous_suffix, feedback_payload)
    return _run_stage(cfg, "refinement", messages, spec, question, observation, token_counter, REFINEMENT_PROMPT_VERSION)

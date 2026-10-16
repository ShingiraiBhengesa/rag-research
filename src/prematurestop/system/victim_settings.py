"""Pinned victim models and generation settings (identical to canonical production).

Extracted unchanged from the archived adaptive-cue pilot configuration. These are
the exact Hugging Face revisions and decoding settings the canonical runs used.
"""

from __future__ import annotations

from types import MappingProxyType

from prematurestop.system.models import FINAL_MODEL_IDS, PIPELINE_VALIDATION_ONLY_MODEL_IDS

VICTIM_MODELS = MappingProxyType(
    {
        "qwen14b": {"model_id": FINAL_MODEL_IDS["qwen14b"], "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"},
        "qwen7b": {"model_id": PIPELINE_VALIDATION_ONLY_MODEL_IDS["qwen7b"], "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
        "llama8b": {"model_id": FINAL_MODEL_IDS["llama8b"], "revision": "0e9e39f249a16976918f6564b8830bc894c89659"},
    }
)


VICTIM_TOOL_PARSERS = MappingProxyType(
    {"qwen14b": "hermes", "qwen7b": "hermes", "llama8b": "llama3_json"}
)


VICTIM_TEMPERATURE = 0.0


VICTIM_SEED = 20260820


VICTIM_MAX_MODEL_LEN = 32768


RESEARCH_MAX_TOKENS = 1200


COMPRESSION_MAX_TOKENS = 2400


FINAL_REPORT_MAX_TOKENS = 1200


EMERGENCY_MAX_REACT_TOOL_CALLS = 30


MAX_RESEARCHER_ITERATIONS = 6


TOP_K = 3

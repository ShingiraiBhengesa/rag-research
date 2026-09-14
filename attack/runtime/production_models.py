"""Frozen final-production model registry."""

from __future__ import annotations

from types import MappingProxyType


FINAL_PRODUCTION_MODELS = ("qwen14b", "gemma27b", "gpt_oss_20b", "llama8b")

FINAL_MODEL_IDS = MappingProxyType(
    {
        "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
        "gemma27b": "google/gemma-3-27b-it",
        "gpt_oss_20b": "openai/gpt-oss-20b",
        "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    }
)

PIPELINE_VALIDATION_ONLY_MODEL_IDS = MappingProxyType(
    {
        "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    }
)

MODEL_TOOL_CALL_PARSERS = MappingProxyType(
    {
        "qwen14b": "hermes",
        "gemma27b": "functiongemma",
        "gpt_oss_20b": "openai",
        "llama8b": "llama3_json",
    }
)

MODEL_REASONING_PARSERS = MappingProxyType(
    {
        "gpt_oss_20b": "openai_gptoss",
    }
)

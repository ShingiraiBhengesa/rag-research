"""Canonical analysis package for completed Qwen and Llama production traces.

Gemma27b and GPT-OSS-20B are excluded at the model-family level throughout
this package (see config.EXCLUDED_MODEL_FAMILIES), for different, unrelated
reasons: GPT-OSS-20B's production matrix is still blocked (see repo-root
PUBLISHING.md); gemma27b's first production run used a misconfigured vLLM
tool-call parser and is quarantined pending a fresh re-run (see
analysis/CHANGELOG.md). This package never opens a trace, result, or job
file for an excluded model, and never writes into its production
directories.
"""

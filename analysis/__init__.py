"""Canonical analysis package for completed Qwen, Llama, and Gemma production traces.

GPT-OSS-20B is excluded at the model-family level throughout this package
(see config.EXCLUDED_MODEL_FAMILIES; its production matrix is still blocked,
see repo-root PUBLISHING.md). This package never opens a GPT-OSS-20B trace,
result, or job file, and never writes into its production directories.
"""

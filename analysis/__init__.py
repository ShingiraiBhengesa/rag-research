"""Canonical analysis package for completed Qwen, Llama, Gemma, and GPT-OSS
production traces.

All 4 models are in config.INCLUDED_MODELS. Gemma27b's fresh, corrected
re-run (see analysis/CHANGELOG.md) is complete across all 4 datasets.
GPT-OSS-20B is complete for 3 of 4 datasets (hotpot, musique,
multihoprag) -- 2wiki remains at 0/108 shards pending a separate,
already-built vLLM harmony-parser fix; any code iterating
config.DATASETS for gpt_oss_20b must tolerate 2wiki being entirely
absent. See config.MODEL_SCOPE_STATEMENT for the full history. The
qwen_llama_results_bundle/ deliverable remains locked to exactly
Qwen14B/Llama8B regardless of this module's INCLUDED_MODELS (see
analysis/canonical/results_bundle/scope.py) -- gemma27b/gpt_oss_20b
results live in a separate bundle.
"""

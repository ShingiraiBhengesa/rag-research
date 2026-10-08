"""Explicit, locked model/dataset scope for this results bundle.

Mirrors `analysis.canonical.results_bundle.scope`'s reasoning: pin this
bundle's scope explicitly rather than reading `analysis.config.INCLUDED_MODELS`
at generation time, so it doesn't silently drift if that repo-wide setting
changes again for a reason unrelated to gemma27b/gpt_oss_20b.

Unlike the Qwen/Llama bundle, this bundle's model coverage is asymmetric:
gpt_oss_20b's 2wiki dataset is not just "not yet run" but actively blocked
(a vLLM harmony-format parser crash distinct from anything that affected
gemma27b -- see analysis/config.py's MODEL_SCOPE_STATEMENT for the full
history). `MODEL_DATASETS` records this explicitly per model rather than
assuming every model in `EXTENDED_BUNDLE_MODELS` has every dataset in
`analysis.config.DATASETS` -- every generator in this package must iterate
`MODEL_DATASETS[model]`, never `config.DATASETS` directly, when looping
per-model.
"""

from __future__ import annotations

from analysis import config

EXTENDED_BUNDLE_MODELS: tuple[str, ...] = ("gemma27b", "gpt_oss_20b")

MODEL_DATASETS: dict[str, tuple[str, ...]] = {
    "gemma27b": config.DATASETS,
    "gpt_oss_20b": ("hotpot", "musique", "multihoprag"),
}

# The one dataset deliberately excluded from this bundle's analysis, and
# why -- kept as an explicit constant (not just absence from MODEL_DATASETS)
# so every report/doc can cite the same string instead of re-deriving it.
GPT_OSS_20B_2WIKI_STATUS = (
    "gpt_oss_20b/2wiki: 0/108 shards. Blocked by a vLLM harmony-format "
    "parsing crash (APIError('unexpected tokens remaining in message "
    "header: ...')), root-caused to vLLM's own server-side harmony "
    "token-processing loop (not a swappable client-side tool parser). A "
    "fix exists (attack/runtime/gptoss_lenient_harmony_parser.py, unit "
    "tested against the real observed malformed patterns), but is "
    "deliberately not yet wired into production or resubmitted. Excluded "
    "from this bundle's analysis entirely, not imputed or estimated."
)

assert set(MODEL_DATASETS) == set(EXTENDED_BUNDLE_MODELS)
assert MODEL_DATASETS["gemma27b"] == config.DATASETS
assert set(MODEL_DATASETS["gpt_oss_20b"]) == set(config.DATASETS) - {"2wiki"}

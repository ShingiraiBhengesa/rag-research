"""Explicit, locked model scope for this results bundle.

`prematurestop.evaluation.config.INCLUDED_MODELS` is a repo-wide, mutable setting that
changes as other models enter or leave production independently of this
bundle -- it has already moved twice in this repo's history (gemma27b
entered, then was re-excluded after a vLLM tool-call-parser
misconfiguration was found; see `src/prematurestop/evaluation/config.py`'s
`GEMMA_GPT_EXCLUSION_STATEMENT`). This results bundle's committed scope
must not move when that repo-wide setting changes for a reason unrelated to
Qwen14B/Llama8B, so it is pinned here explicitly rather than read from
`config.INCLUDED_MODELS` at generation time.

Every canonical output in `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/` (this
package) and `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/` (which shares the same
frozen scope, see that package's own README.md) is built against exactly
these two models, regardless of what `config.INCLUDED_MODELS` says.
"""

from __future__ import annotations

CANONICAL_BUNDLE_MODELS: tuple[str, ...] = ("qwen14b", "llama8b")

# The same bundle machinery also builds the Gemma27B/GPT-OSS-20B bundle. That
# is an explicit, per-run choice (`build_bundle --models gemma27b,gpt_oss_20b`),
# never a change to the locked default above: every stage reads the active
# scope through `bundle_models()`, and only `set_bundle_models()` changes it.
_active_models: tuple[str, ...] = CANONICAL_BUNDLE_MODELS


def bundle_models() -> tuple[str, ...]:
    """Models the current bundle run covers (default: CANONICAL_BUNDLE_MODELS)."""
    return _active_models


def set_bundle_models(models) -> None:
    """Select the models for this bundle run. Called once by build_bundle.main()."""
    global _active_models
    models = tuple(models)
    if not models:
        raise ValueError("bundle model scope must not be empty")
    _active_models = models

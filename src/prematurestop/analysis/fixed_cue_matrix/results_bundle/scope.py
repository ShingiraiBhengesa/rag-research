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

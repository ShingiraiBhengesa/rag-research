"""Qwen14B/Llama8B frozen results-bundle generator.

Consumes the existing, already-validated `analysis/` package (primarily
`analysis.canonical.canonical_completion`, `analysis.evidence`,
`analysis.answer_evaluation`, `analysis.canonical.retrieval_equivalence`)
and adds exactly the extension logic needed to produce the frozen
`qwen_llama_results_bundle/` deliverable: full step-1-29 PCR/route curves,
paired deltas with confidence intervals, a unified condition-comparison
manifest, process/recovery metrics, and validation gates.

This package never modifies the modules it imports from -- see
`/home/sbhengesa/.claude/plans/robust-spinning-lerdorf.md` for the full
design rationale (why `canonical_completion.py` is the sole PCR/route
foundation, not the legacy `analysis/pcr.py`).
"""

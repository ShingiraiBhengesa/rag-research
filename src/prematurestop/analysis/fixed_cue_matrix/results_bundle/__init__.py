"""Qwen14B/Llama8B frozen results-bundle generator.

Consumes the existing, already-validated `analysis/` package (primarily
`prematurestop.evaluation.completion`, `prematurestop.evaluation.evidence`,
`prematurestop.evaluation.answer_evaluation`, `prematurestop.analysis.fixed_cue_matrix.audits.retrieval_equivalence`)
and adds exactly the extension logic needed to produce the frozen
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/` deliverable: full step-1-29 PCR/route curves,
paired deltas with confidence intervals, a unified condition-comparison
manifest, process/recovery metrics, and validation gates.

This package never modifies the modules it imports from -- see
the project's internal planning notes for the full
design rationale (why `canonical_completion.py` is the sole PCR/route
foundation, not the legacy `src/prematurestop/evaluation/pcr.py`).
"""

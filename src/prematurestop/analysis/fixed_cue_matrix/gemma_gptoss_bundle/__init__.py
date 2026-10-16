"""Gemma27B/GPT-OSS-20B detailed results bundle generator.

Sibling to `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/` (the Qwen14B/Llama8B
bundle), reusing the same canonical machinery
(`prematurestop.evaluation.completion`, `prematurestop.evaluation.evidence`,
`prematurestop.evaluation.answer_evaluation`, `prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons`,
and most of `prematurestop.analysis.fixed_cue_matrix.results_bundle.gates`) rather than
reimplementing it. This package never modifies the modules it imports from,
and never touches `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` or
`src/prematurestop/analysis/fixed_cue_matrix/results_bundle/`'s own locked `CANONICAL_BUNDLE_MODELS`
scope -- see this package's own `scope.py`.

Model scope is asymmetric by design, not a bug to work around: gemma27b is
complete across all 4 datasets; gpt_oss_20b is complete for 3 of 4
(hotpot, musique, multihoprag) -- its 2wiki dataset is at 0/108 shards,
blocked by a separate, already-diagnosed vLLM harmony-parser crash
(src/prematurestop/system/tool_parsers/gptoss_lenient_harmony.py exists and is tested, but
deliberately not yet deployed). Every module here is written to report that
gap explicitly rather than silently omit it.
"""

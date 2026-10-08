"""Gemma27B/GPT-OSS-20B detailed results bundle generator.

Sibling to `analysis/canonical/results_bundle/` (the Qwen14B/Llama8B
bundle), reusing the same canonical machinery
(`analysis.canonical.canonical_completion`, `analysis.evidence`,
`analysis.answer_evaluation`, `analysis.canonical.results_bundle.comparisons`,
and most of `analysis.canonical.results_bundle.gates`) rather than
reimplementing it. This package never modifies the modules it imports from,
and never touches `qwen_llama_results_bundle/` or
`analysis/canonical/results_bundle/`'s own locked `CANONICAL_BUNDLE_MODELS`
scope -- see this package's own `scope.py`.

Model scope is asymmetric by design, not a bug to work around: gemma27b is
complete across all 4 datasets; gpt_oss_20b is complete for 3 of 4
(hotpot, musique, multihoprag) -- its 2wiki dataset is at 0/108 shards,
blocked by a separate, already-diagnosed vLLM harmony-parser crash
(attack/runtime/gptoss_lenient_harmony_parser.py exists and is tested, but
deliberately not yet deployed). Every module here is written to report that
gap explicitly rather than silently omit it.
"""

# Layout migration, 2026-10-16

Every file moved with `git mv`; `git log --follow` works. Directory moves first, then individual modules.

## Directories

| Old | New |
|---|---|
| `open_deep_research` | `third_party/open_deep_research` |
| `experiments/dynamic_bridge_cue_pilot` | `experiments/whitebox/e2_bridge_mechanism` |
| `qwen_llama_results_bundle` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama` |
| `gemma_gptoss_results_bundle` | `results/whitebox/e1_fixed_cue_matrix/gemma_gptoss` |
| `analysis/canonical/results_bundle` | `src/prematurestop/analysis/fixed_cue_matrix/results_bundle` |
| `analysis/canonical/gemma_gptoss_bundle` | `src/prematurestop/analysis/fixed_cue_matrix/gemma_gptoss_bundle` |
| `analysis/canonical/final_gap_audit` | `src/prematurestop/analysis/fixed_cue_matrix/audits/final_gap_audit` |
| `analysis/final_paper_evaluation` | `src/prematurestop/analysis/fixed_cue_matrix/paper_evaluation` |
| `analysis/evidence_hop_coverage` | `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage` |
| `analysis/evidence_coverage_distributions_v1` | `src/prematurestop/analysis/fixed_cue_matrix/evidence_distributions` |
| `attack/dynamic_bridge_pilot/prompts` | `src/prematurestop/cues/dynamic/prompts` |
| `attack/adaptive_pilot, analysis/canonical/reorg_prep, presentation_export` | `archive/ (local only, not published)` |

## Modules

| Old | New |
|---|---|
| `analysis/answer_evaluation.py` | `src/prematurestop/evaluation/answer_evaluation.py` |
| `analysis/canonical/canonical_completion.py` | `src/prematurestop/evaluation/completion.py` |
| `analysis/canonical/reconcile.py` | `src/prematurestop/analysis/fixed_cue_matrix/audits/reconcile.py` |
| `analysis/canonical/retrieval_equivalence.py` | `src/prematurestop/analysis/fixed_cue_matrix/audits/retrieval_equivalence.py` |
| `analysis/canonical/test_canonical_completion.py` | `tests/evaluation/test_completion.py` |
| `analysis/cli.py` | `src/prematurestop/analysis/fixed_cue_matrix/cli.py` |
| `analysis/completion_inventory.py` | `src/prematurestop/evaluation/completion_inventory.py` |
| `analysis/config.py` | `src/prematurestop/evaluation/config.py` |
| `analysis/evidence.py` | `src/prematurestop/evaluation/evidence.py` |
| `analysis/pairing.py` | `src/prematurestop/evaluation/pairing.py` |
| `analysis/pcr.py` | `src/prematurestop/evaluation/pcr.py` |
| `analysis/repr_parsing.py` | `src/prematurestop/evaluation/repr_parsing.py` |
| `analysis/statistics.py` | `src/prematurestop/stats/statistics.py` |
| `analysis/tests/test_config_guards.py` | `tests/evaluation/test_config_guards.py` |
| `analysis/tests/test_evidence.py` | `tests/evaluation/test_evidence.py` |
| `analysis/tests/test_final_paper_evaluation.py` | `tests/evaluation/test_final_paper_evaluation.py` |
| `analysis/tests/test_pcr.py` | `tests/evaluation/test_pcr.py` |
| `analysis/tests/test_repr_parsing.py` | `tests/evaluation/test_repr_parsing.py` |
| `analysis/tests/test_results_bundle_scope.py` | `tests/evaluation/test_results_bundle_scope.py` |
| `analysis/tests/test_trace_parser.py` | `tests/evaluation/test_trace_parser.py` |
| `analysis/trace_parser.py` | `src/prematurestop/evaluation/trace_parser.py` |
| `analysis/trace_schema.py` | `src/prematurestop/evaluation/trace_schema.py` |
| `attack/answer_scoring/answer_evaluator_v2.py` | `src/prematurestop/evaluation/answer_scoring.py` |
| `attack/cue_definitions.py` | `src/prematurestop/cues/fixed/families.py` |
| `attack/dynamic_bridge_pilot/analysis.py` | `src/prematurestop/pipelines/bridge_mechanism/analysis.py` |
| `attack/dynamic_bridge_pilot/audit.py` | `src/prematurestop/pipelines/bridge_mechanism/audit.py` |
| `attack/dynamic_bridge_pilot/bridge.py` | `src/prematurestop/cues/dynamic/bridge.py` |
| `attack/dynamic_bridge_pilot/cli.py` | `src/prematurestop/pipelines/bridge_mechanism/cli.py` |
| `attack/dynamic_bridge_pilot/config.py` | `src/prematurestop/pipelines/bridge_mechanism/config.py` |
| `attack/dynamic_bridge_pilot/cues.py` | `src/prematurestop/cues/dynamic/generation.py` |
| `attack/dynamic_bridge_pilot/entail.py` | `src/prematurestop/cues/dynamic/entail.py` |
| `attack/dynamic_bridge_pilot/freeze.py` | `src/prematurestop/pipelines/bridge_mechanism/freeze.py` |
| `attack/dynamic_bridge_pilot/lambda_fit.py` | `src/prematurestop/cues/dynamic/lambda_fit.py` |
| `attack/dynamic_bridge_pilot/llm.py` | `src/prematurestop/cues/dynamic/llm.py` |
| `attack/dynamic_bridge_pilot/observation.py` | `src/prematurestop/cues/dynamic/observation.py` |
| `attack/dynamic_bridge_pilot/outcomes.py` | `src/prematurestop/pipelines/bridge_mechanism/outcomes.py` |
| `attack/dynamic_bridge_pilot/paths.py` | `src/prematurestop/pipelines/bridge_mechanism/paths.py` |
| `attack/dynamic_bridge_pilot/runner.py` | `src/prematurestop/pipelines/bridge_mechanism/runner.py` |
| `attack/dynamic_bridge_pilot/selection_report.py` | `src/prematurestop/pipelines/bridge_mechanism/selection_report.py` |
| `attack/dynamic_bridge_pilot/splits.py` | `src/prematurestop/pipelines/bridge_mechanism/splits.py` |
| `attack/dynamic_bridge_pilot/store.py` | `src/prematurestop/infra/store.py` |
| `attack/dynamic_bridge_pilot/tests/test_pilot.py` | `tests/pipelines/test_bridge_mechanism.py` |
| `attack/dynamic_bridge_pilot/unsupported.py` | `src/prematurestop/cues/dynamic/unsupported.py` |
| `attack/launch/generate_production_arrays.py` | `src/prematurestop/pipelines/fixed_cue_matrix/launch.py` |
| `attack/metrics.py` | `src/prematurestop/stats/metrics.py` |
| `attack/runtime/benchmark_search.py` | `src/prematurestop/system/benchmark_search.py` |
| `attack/runtime/corpus.py` | `src/prematurestop/data/corpus.py` |
| `attack/runtime/cues.py` | `src/prematurestop/cues/fixed/registry.py` |
| `attack/runtime/evidence.py` | `src/prematurestop/data/evidence.py` |
| `attack/runtime/gemma3_tool_code_parser.py` | `src/prematurestop/system/tool_parsers/gemma3_tool_code.py` |
| `attack/runtime/gptoss_lenient_harmony_parser.py` | `src/prematurestop/system/tool_parsers/gptoss_lenient_harmony.py` |
| `attack/runtime/hashing.py` | `src/prematurestop/infra/hashing.py` |
| `attack/runtime/musique_scope_audit.py` | `src/prematurestop/data/musique_scope_audit.py` |
| `attack/runtime/production_manifest.py` | `src/prematurestop/data/manifests.py` |
| `attack/runtime/production_models.py` | `src/prematurestop/system/models.py` |
| `attack/runtime/production_runner.py` | `src/prematurestop/pipelines/fixed_cue_matrix/production_runner.py` |
| `attack/runtime/runner.py` | `src/prematurestop/system/runner.py` |
| `attack/runtime/runtime_gate.py` | `src/prematurestop/pipelines/fixed_cue_matrix/runtime_gate.py` |
| `attack/runtime/trace_schema.py` | `src/prematurestop/system/trace_schema.py` |
| `attack/tests/test_answer_evaluator_v2.py` | `tests/evaluation/test_answer_evaluator_v2.py` |
| `attack/tests/test_gemma3_tool_code_parser.py` | `tests/system/test_gemma3_tool_code_parser.py` |
| `attack/tests/test_gptoss_lenient_harmony_parser.py` | `tests/system/test_gptoss_lenient_harmony_parser.py` |
| `attack/tests/test_v3_infra.py` | `tests/system/test_v3_infra.py` |
| `attack/tests/test_v3_musique.py` | `tests/data/test_v3_musique.py` |
| `attack/tests/test_v3_musique_real_schema.py` | `tests/data/test_v3_musique_real_schema.py` |
| `attack/tests/test_v3_production.py` | `tests/pipelines/test_v3_production.py` |

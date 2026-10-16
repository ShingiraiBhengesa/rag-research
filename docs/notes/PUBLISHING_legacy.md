# Publishing Checklist

This checklist keeps the final production experiment publishable without
disturbing queued production jobs.

## ⚠️ ACTION NEEDED: gpt_oss_20b production jobs are permanently stuck (found 2026-09-11, being fixed 2026-09-14)

All 4 `gpt_oss_20b` production array jobs (`v3_{hotpot,2wiki,musique,multihoprag}_gpt_oss_20b`)
carry an `afterok:149854_*` dependency on `v3_recovery_gate_gpt_oss_20b`, which **FAILED**
on 2026-09-08 (`sacct -j 149854`: exit code 20:0) and was never resubmitted. An `afterok`
dependency on a failed job can never be satisfied (`squeue` reason: `DependencyNeverSatisfied`)
-- these 432 array tasks will sit in `PENDING` forever without human action. This was
discovered read-only (no job touched) while building the final paper analysis, and is still
unresolved.

Investigation as of 2026-09-14: the gate ran 2 dev examples. `NaturalClean` failed with a
genuinely empty final-report response from gpt-oss-20b (not a false-negative extraction bug
in the gate, as originally suspected below) despite ODR's built-in empty-response retry
logic (up to 3 retries) being exhausted. `Natural_PreRegisteredSufficiencySaturation`, run
against the SAME dev example, produced a full, well-formed report. This looks like an
intermittent (not universal) empty-final-report failure mode specific to gpt-oss-20b, which
is a real concern given the ~2-week compute commitment a full resubmission represents --
such a failure, if it recurs during production, would NOT surface as a row-level error (ODR's
`final_report_generation` catches it internally and returns a normally-structured "successful"
row containing garbage/empty content instead of failing loudly).

Not yet decided: whether to (a) re-run the gate as-is and treat a pass as sufficient
confidence, (b) harden the retry/token-budget behavior in
`open_deep_research/src/open_deep_research/deep_researcher.py`'s final-report path before
resubmitting, or (c) accept the intermittent-failure risk and resubmit, relying on
downstream analysis to detect and quarantine any garbage rows post hoc. This decision, plus
actually fixing the dependency and resubmitting the 4 arrays, is the one remaining step
before all four models are complete.

`gemma27b` has a separate, now-fixed problem of its own -- see the next section.

## RESOLVED: gemma27b's first production run used a misconfigured tool-call parser (found and fixed 2026-09-14)

`gemma27b` fully completed 432/432 shards across all 4 datasets on 2026-09-13, but its
completion data showed ~99% "no tool call" episodes (driven through ODR's
`force_benchmark_search` safety net instead of real model decisions). Root-caused to
`src/prematurestop/system/models.py` configuring `--tool-call-parser functiongemma` for
`google/gemma-3-27b-it` -- `functiongemma` is vLLM's parser for a completely different,
unrelated model (`google/functiongemma-270m-it`, confirmed via that parser's own module
docstring), which never recognizes `gemma-3-27b-it`'s actual
` ```tool_code\n<call>\n``` `-fenced tool calls. No vLLM version ships native Gemma-3
`tool_code` support (both upstream feature requests, vllm-project/vllm#15403 and #16482,
were closed "not planned").

Fixed by writing a custom vLLM tool-parser plugin,
`src/prematurestop/system/tool_parsers/gemma3_tool_code.py`, registered via `--tool-parser-plugin` (see
`src/prematurestop/system/models.py`'s `MODEL_TOOL_CALL_PARSERS["gemma27b"] =
"gemma3_tool_code"` and `src/prematurestop/pipelines/fixed_cue_matrix/launch.py`'s
`tool_parser_plugin` field). Handles both observed malformed-call sub-formats (a Python-call
expression and a JSON object) and normalizes argument names/shapes to the real tool schemas
(`tavily_search(queries: list[str])`, `think_tool(reflection: str)`) since Gemma's calls
don't match those exactly either. Unit-tested against real captured trace text
(`tests/system/test_gemma3_tool_code_parser.py`, run under the `vllm` conda env where the
`vllm` package is importable).

A real bug was found only through live validation, not the unit tests alone: this pipeline
runs ODR under LangGraph's `.astream_events()`, which makes the underlying chat-model calls
stream at the token level even though each node just does a plain `ainvoke()` (confirmed:
`on_chat_model_stream` fires ~2000x per generation in real traces). vLLM therefore calls the
parser's *streaming* extraction path, not the non-streaming one -- the first version of the
plugin only implemented the non-streaming path correctly, so the first live gate run (job
151760) still showed 100% forced-search fallback despite passing unit tests. Fixed by
properly implementing `extract_tool_calls_streaming` (buffers until a complete
` ```tool_code``` ` block closes, withholding partial markers split across chunk boundaries,
then emits the full tool call in one shot). Confirmed working end-to-end against a real
vLLM+gemma-3-27b-it endpoint (job 151777, gate `RUNTIME_GATE_PASSED.json`, and direct
inspection of the resulting trace: genuine `tavily_search`/`think_tool` tool calls with real
`chatcmpl-tool-*` ids, zero `force_benchmark_search` occurrences).

The broken run's data (`results/traces/gemma27b/`, `results/runs/gemma27b/`, the
`gemma27b_direct_exit_content_audit.csv` audit, the `all_models_20260913` canonical outputs,
and the `v3_final_completed_qwen_llama_gemma_20260913` evaluation directory) is quarantined
under `archive/legacy_results/gemma27b_broken_functiongemma_parser_run_20260913/` --
kept, not deleted. `gemma27b` was moved back from `src/prematurestop/evaluation/config.py`'s `INCLUDED_MODELS`
into `EXCLUDED_MODEL_FAMILIES` pending a fresh run. All 4 gemma27b production arrays were
resubmitted 2026-09-14 (jobs 151778-151781) with the fix; re-add gemma27b to
`INCLUDED_MODELS` once that run completes.

**Fixing the parser does not retroactively validate the quarantined run.**
The corrected parser (`src/prematurestop/system/tool_parsers/gemma3_tool_code.py`, unit-tested
and confirmed working live against a real endpoint) only makes a *future*
gemma27b run trustworthy -- it does not change the fact that the already-run
432-shard batch above used the broken parser and remains invalid. No
gemma27b experimental finding should be reported until a full gemma27b
production run completes under the corrected parser and is independently
verified (completion-inventory + validation gates), the same bar the
qwen14b/llama8b canonical bundle already meets. The canonical thesis
results bundle (`results/whitebox/e1_fixed_cue_matrix/qwen_llama/`) remains Qwen14B and Llama8B
only; it does not include gemma27b in any form.

## Branch state

Current working branch during cleanup: `paper-pcr40-results`.

Before merging to `main`, verify that the final production jobs have completed
or that any remaining queued jobs are intentionally preserved against the exact
source tree being merged.

## Repository reorganization (completed 2026-09-13/14)

The codebase has been fully restructured for publication readiness, in two
passes:

1. **2026-09-13**: `whitebox_attack/v3/`'s flat ~25-script directory split
   into `runtime/`, `launch/`, `legacy_evaluators/`, `dev_tools/`
   subpackages (analysis code kept nested at the time).
2. **2026-09-14**: full top-level flattening, dropping the `whitebox_`/`v3`
   naming entirely since there is only one version of this experiment:
   - `whitebox_attack/v3/*` → top-level `attack/*`
   - `whitebox_attack/v3/analysis/` → top-level `analysis/`
   - `whitebox_paper_results/e2e_full_system_v3/production/` → top-level
     `results/`
   - all pre-v3/legacy code and superseded result directories
     (`whitebox_attack/archive/pre_v3_root_20260910/`, `e2e_full_system_v1/v2`,
     `pcr_full_completion_v1`/`v2_60`, `e2e_final_answer_v1`, the spurious
     duplicate `whitebox_attack/whitebox_paper_results/`) → top-level
     `archive/`, never deleted.

Both passes were dry-run before executing, produced zero test regressions
(full 178-test suite), and preserved historical launch-time artifacts (see
"Provenance preservation" below). The scripts that performed both moves are
kept as a permanent record at `analysis/canonical/reorg_prep/`. Also fixed as
part of this reorganization: the trace-parser's forced-search branch-point
gap (see `docs/CHANGELOG_analysis.md`) and a stale-`trace_path`-inside-persisted-data
issue (see `docs/notes/analysis_package_README_legacy.md`, key empirical fact 7) -- neither is
cosmetic, both affect correctness of completion/route claims. (Separately,
gemma27b's own vLLM tool-call-parser misconfiguration was found and fixed
2026-09-14 -- see the dedicated section above.)

This checklist's older sections below (pre-2026-09-14) refer to the old
`whitebox_attack/`/`whitebox_paper_results/` paths; they are kept as a
historical record of what was already done, not as current instructions.
Current paths are `attack/`, `analysis/`, `results/`, `archive/` throughout.

## Publishable final artifacts

Track or archive with the paper:

- `attack/`
- `analysis/`
- `open_deep_research/src/open_deep_research/`
- `results/whitebox/e1_fixed_cue_matrix/production_records/matrix/v3_final_production_matrix.json`
- `results/whitebox/e1_fixed_cue_matrix/production_records/prelaunch/PRELAUNCH_REPORT.json`
- `results/submissions/submission_*.json`
- `results/evaluation/v3_final_completed_qwen_llama_20260914/` (2-model,
  qwen14b+llama8b only -- gemma27b is excluded again pending its re-run, see
  above) -- only `pcr/`, `statistics/`, `by_model_dataset/`, and
  `answers/final_answer_summary.csv` should be tracked (mirror the
  `.gitignore` pattern already in place for the superseded
  `v3_final_completed_qwen_llama_gemma_20260913/` directory once this one is
  generated: raw per-row predictions/evidence/audits are reproducible from
  traces + code via `prematurestop.analysis.fixed_cue_matrix.cli`, not paper deliverables).
- `results/evaluation/final_production/` (regenerate once `gpt_oss_20b` completes)

Do not cite all-four-model aggregate values until both `gemma27b`'s
re-run and `gpt_oss_20b` have completed and `prematurestop.analysis.fixed_cue_matrix.cli --stage all` has
been rerun with all four models included.

## Provenance preservation

`results/whitebox/e1_fixed_cue_matrix/production_records/matrix/v3_final_production_matrix.json` and
`results/whitebox/e1_fixed_cue_matrix/production_records/prelaunch/PRELAUNCH_REPORT.json` are the original files from the
actual production launch. Both `--mode generate`/`--mode validate` re-runs
performed during the 2026-09-13 and 2026-09-14 reorganizations would have
overwritten them in place; each time, the freshly-regenerated version was
instead saved alongside under a dated suffix
(`_POST_REORG_REGENERATION_20260914.json`,
`_POST_REORG_REVALIDATION_20260914.json`,
`_POST_TOPLEVEL_REORG_REGENERATION_20260914.json`,
`_POST_TOPLEVEL_REORG_REVALIDATION_20260914.json`), and the true original was
restored from `chapter3_methodology_audit/chapter3_methodology_bundle.zip`.
This proves both reorganizations were behavior-preserving. If either file is
regenerated again (e.g. after `gpt_oss_20b` completes), follow the same
preserve-then-restore pattern before overwriting the historical original.

## Validation-only or local-only artifacts

Keep out of final claims:

- qwen7b outputs and sbatch scripts (`results/quarantine/qwen7b_pipeline_validation_20260828_174947/`)
- qwen14b pre-freeze partial outputs (`results/quarantine/qwen14b_prefreeze_20260828_174045/`)
- runtime logs, raw traces, Triton/xgrammar caches, and per-shard scratch files
- early exploratory attack-design result files, superseded by the current
  attack (now under `archive/`) unless used in an appendix

## Active-Job Safety

Queued SLURM jobs import source files from this working tree at runtime.
While `gpt_oss_20b` or `gemma27b` jobs are queued or running, do not edit,
rename, delete, or reformat:

- `src/prematurestop/system/models.py`
- `src/prematurestop/data/manifests.py`
- `src/prematurestop/cues/fixed/registry.py`
- `src/prematurestop/system/benchmark_search.py`
- `src/prematurestop/data/corpus.py`
- `src/prematurestop/data/evidence.py`
- `src/prematurestop/infra/hashing.py`
- `src/prematurestop/system/trace_schema.py`
- `src/prematurestop/system/runner.py`
- `src/prematurestop/pipelines/fixed_cue_matrix/runtime_gate.py`
- `src/prematurestop/pipelines/fixed_cue_matrix/production_runner.py`
- `src/prematurestop/system/tool_parsers/gemma3_tool_code.py`
- `src/prematurestop/pipelines/fixed_cue_matrix/launch.py`
- `src/prematurestop/stats/metrics.py`, `src/prematurestop/cues/fixed/families.py`
- `open_deep_research/src/open_deep_research/configuration.py`
- `open_deep_research/src/open_deep_research/deep_researcher.py`
- `open_deep_research/src/open_deep_research/prompts.py`
- `open_deep_research/src/open_deep_research/state.py`
- `open_deep_research/src/open_deep_research/utils.py`

The `analysis/` package, all documentation, and `archive/` are safe to edit
at any time -- none are imported by any SLURM job.

## Remaining steps to full publication

1. Wait for all 432 `gemma27b` shards (jobs 151778-151781, resubmitted
   2026-09-14 with the parser fix) to complete (`squeue`/shard-count checks;
   no polling faster than the job's own pace).
2. Decide how to handle the gpt_oss_20b gate's intermittent empty-final-report
   failure (see the ACTION NEEDED section above).
3. Fix the dead `afterok` dependency and resubmit the 4 `gpt_oss_20b`
   production arrays.
4. Wait for all 432 `gpt_oss_20b` shards to complete.
5. Once gemma27b's re-run completes: re-add `"gemma27b"` to
   `src/prematurestop/evaluation/config.py`'s `INCLUDED_MODELS` (remove it from
   `EXCLUDED_MODEL_FAMILIES`/`_EXCLUDED_PATH_TOKENS`), then once gpt_oss_20b
   also completes, extend further to include it too. Re-run
   `python -m prematurestop.analysis.fixed_cue_matrix.cli --stage all` into a new output directory (e.g.
   `results/evaluation/v3_final_completed_all_models_<date>/`) -- never
   overwrite a prior evaluation directory in place; mark it superseded in
   the new directory instead.
6. Regenerate/finalize paper tables from the all-models run.
7. Optionally prune `results/quarantine/`/`archive/` further, or leave as
   forensic history -- low urgency.
8. Commit in a few focused commits (gate fix + resubmission; analysis
   model extension; final tables) rather than one giant one.

## Historical record: 2026-09-10 cleanup (completed while jobs were still active, pre-dates the reorg above)

These touched only dead-weight paths never imported by the active `v3_*`
SLURM jobs (verified via `squeue`/`scontrol show job` plus grepping every
`.sbatch` script and all of `whitebox_attack/`/`open_deep_research/` for
references before moving anything). Paths below are as they were at the
time; see the reorganization section above for where everything lives now.

- Deleted untracked vendored/legacy top-level dirs unrelated to this
  project's ODR/whitebox-attack line of work: `adaptive-rag/`, `self-rag/`,
  `attack/`, `results/`, `scripts/`, and a stray `=0.12.0` file.
- `git rm -r`'d the legacy `whitebox_results/` directory (fully superseded
  by `whitebox_paper_results/`).
- Moved the ~20 dead pre-v3 top-level `whitebox_attack/*.py` scripts, plus
  `whitebox_attack/sbatch/` and `whitebox_attack/logs/`, into
  `whitebox_attack/archive/pre_v3_root_20260910/` (now under `archive/`).
  Kept `metrics.py` and `cue_definitions.py` in place at the time -- they
  were live imports of `v3/production_runner.py` and `v3/cues.py`
  respectively (now `src/prematurestop/stats/metrics.py` and `src/prematurestop/cues/fixed/families.py`).
- Moved the 14 superseded one-off `evaluation/` output directories into
  `evaluation/_archive_superseded_20260910/` (now under `archive/`).
- Verified afterward: analysis package still imported and its tests still
  passed, `v3.production_runner`/`v3.cues` still imported cleanly, the
  active jobs' actual sbatch scripts were untouched, and `squeue` showed
  identical job states before and after.

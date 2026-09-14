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

`gemma27b` has no equivalent problem -- it fully completed all 432/432 shards across all 4
datasets on 2026-09-13.

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
part of this reorganization: gemma27b's tool-calling failure mode (see
`analysis/CHANGELOG.md`) and a stale-`trace_path`-inside-persisted-data issue
(see `analysis/README.md`, key empirical fact 7) -- neither is cosmetic, both
affect correctness of completion/route claims for all three completed models.

This checklist's older sections below (pre-2026-09-14) refer to the old
`whitebox_attack/`/`whitebox_paper_results/` paths; they are kept as a
historical record of what was already done, not as current instructions.
Current paths are `attack/`, `analysis/`, `results/`, `archive/` throughout.

## Publishable final artifacts

Track or archive with the paper:

- `attack/`
- `analysis/`
- `open_deep_research/src/open_deep_research/`
- `results/matrix/v3_final_production_matrix.json`
- `results/prelaunch/PRELAUNCH_REPORT.json`
- `results/submissions/submission_*.json`
- `results/evaluation/v3_final_completed_qwen_llama_gemma_20260913/` -- only
  `pcr/`, `statistics/`, `by_model_dataset/`, and
  `answers/final_answer_summary.csv` are tracked. `answers/final_answer_predictions.jsonl`
  (~99M raw per-row predictions), `evidence/*.csv` (per-example rows), and
  all of `audits/` (row/violation-level audit CSVs) are git-ignored --
  reproducible from traces + code via `analysis.cli`, not paper deliverables.
- `results/evaluation/final_production/` (regenerate once `gpt_oss_20b` completes)

Do not cite all-four-model aggregate values until `gpt_oss_20b` has completed
and `analysis.cli --stage all` has been rerun with all four models included.

## Provenance preservation

`results/matrix/v3_final_production_matrix.json` and
`results/prelaunch/PRELAUNCH_REPORT.json` are the original files from the
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
While `gpt_oss_20b` jobs are queued or running, do not edit, rename, delete,
or reformat:

- `attack/runtime/production_models.py`
- `attack/runtime/production_manifest.py`
- `attack/runtime/cues.py`
- `attack/runtime/benchmark_search.py`
- `attack/runtime/corpus.py`
- `attack/runtime/evidence.py`
- `attack/runtime/hashing.py`
- `attack/runtime/trace_schema.py`
- `attack/runtime/runner.py`
- `attack/runtime/runtime_gate.py`
- `attack/runtime/production_runner.py`
- `attack/launch/generate_production_arrays.py`
- `attack/metrics.py`, `attack/cue_definitions.py`
- `open_deep_research/src/open_deep_research/configuration.py`
- `open_deep_research/src/open_deep_research/deep_researcher.py`
- `open_deep_research/src/open_deep_research/prompts.py`
- `open_deep_research/src/open_deep_research/state.py`
- `open_deep_research/src/open_deep_research/utils.py`

The `analysis/` package, all documentation, and `archive/` are safe to edit
at any time -- none are imported by any SLURM job.

## Remaining steps to full publication

1. Decide how to handle the gpt_oss_20b gate's intermittent empty-final-report
   failure (see the ACTION NEEDED section above).
2. Fix the dead `afterok` dependency and resubmit the 4 `gpt_oss_20b`
   production arrays.
3. Wait for all 432 `gpt_oss_20b` shards to complete (`squeue`/shard-count
   checks; no polling faster than the job's own pace).
4. Extend `analysis/config.py`'s `INCLUDED_MODELS` to include `gpt_oss_20b`
   and re-run `python -m analysis.cli --stage all` into a new output
   directory (e.g. `results/evaluation/v3_final_completed_all_models_<date>/`)
   -- never overwrite `v3_final_completed_qwen_llama_gemma_20260913/` in
   place; mark it superseded in the new directory instead.
5. Regenerate/finalize paper tables from the all-models run.
6. Optionally prune `results/quarantine/`/`archive/` further, or leave as
   forensic history -- low urgency.
7. Commit in a few focused commits (gate fix + resubmission; analysis
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
  respectively (now `attack/metrics.py` and `attack/cue_definitions.py`).
- Moved the 14 superseded one-off `evaluation/` output directories into
  `evaluation/_archive_superseded_20260910/` (now under `archive/`).
- Verified afterward: analysis package still imported and its tests still
  passed, `v3.production_runner`/`v3.cues` still imported cleanly, the
  active jobs' actual sbatch scripts were untouched, and `squeue` showed
  identical job states before and after.

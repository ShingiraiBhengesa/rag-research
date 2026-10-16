# Cleanup change log

This task added files only. No existing file was modified, moved, renamed,
or deleted — per the task's explicit scope-and-safety constraints (do not
modify experiment code, configs, prompts, results, or existing files; do not
delete older evaluator scripts).

## Files created (all under `analysis/canonical/`)

- `pre_cleanup_inventory.md` — git/job/completion-status snapshot taken
  before any other file in this package was written.
- `pre_cleanup_sha256.txt` — SHA-256 of every relevant v3 source file, the 5
  modified `open_deep_research` files, and key production config artifacts,
  at the same snapshot point.
- `pre_cleanup_git_diff.patch` — the exact uncommitted diff (vs. `HEAD`) for
  `open_deep_research/`, `attack/metrics.py`, `.gitignore`, and both
  README files, captured before this task touched anything.
- `legacy_metric_inventory.md` — Phase 2 inventory and classification of
  every existing PCR/completion-adjacent implementation in the repository.
- `condition_metadata.csv` — Phase 8 analysis-facing condition metadata
  (internal identifiers unchanged; only a `thesis_display_label` column is
  new).
- `canonical_completion.py` — the canonical PCR/route-taxonomy
  implementation (Phase 3).
- `test_canonical_completion.py` — 20 unit/regression tests (Phase 5), all
  passing (`python -m unittest ... -v` → `Ran 20 tests ... OK`).
- `retrieval_equivalence.py` — Phase 9D retrieval-equivalence audit script.
- `reconcile.py` — Phase 6 reconciliation script (canonical vs. `pcr.py`).
- `canonical_run_level_completion.csv`, `canonical_pcr_by_condition.csv`,
  `canonical_pcr_by_dataset.csv`, `canonical_pcr_by_model.csv`,
  `canonical_route_decomposition.csv`, `canonical_paired_condition_deltas.csv`,
  `canonical_exclusion_summary.csv` — Phase 4 outputs, computed from real,
  already-completed qwen14b/llama8b production data (43,200 rows).
- `canonical_vs_previous_comparison.md`, `canonical_vs_previous_row_differences.csv`
  — Phase 6 outputs.
- `retrieval_equivalence_by_pair.csv`, `retrieval_equivalence_summary.md` —
  Phase 9D outputs.
- `final_design_matrix.csv`, `final_sample_counts.csv` — Phase 9C/9F outputs.
- `chapter3_final_source_of_truth.md` — Phase 10 final report.
- `cleanup_change_log.md` — this file.

## Files explicitly NOT modified (verified via `git status`/`git diff` and
## SHA-256 comparison against `pre_cleanup_sha256.txt` at task completion)

Every file on the production import-lock list in `PUBLISHING.md`
(`production_models.py`, `production_manifest.py`, `cues.py`,
`benchmark_search.py`, `corpus.py`, `evidence.py`, `hashing.py`,
`trace_schema.py`, `runner.py`, `runtime_gate.py`, `production_runner.py`,
and the 5 modified `open_deep_research` core files), every raw production
result file under `whitebox_paper_results/e2e_full_system_v3/production/`,
every file in the pre-existing `analysis/` package (read
via import only, never edited), and every file already archived under
`whitebox_attack/archive/`.

## No SLURM job was started, restarted, cancelled, or otherwise modified

`squeue` was queried read-only at the start (`pre_cleanup_inventory.md`) and
is queried read-only again for the final report's "remaining incomplete
cells" section. No `sbatch`/`scancel`/`scontrol update` command was ever run.

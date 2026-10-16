# Pre-cleanup reproducibility inventory

Generated 2026-09-11 (UTC timestamps in sibling files), before any file in this
`canonical_completion_v1/` package was written. This snapshot exists so the exact
pre-canonicalization state of the repository can be reconstructed if needed. Nothing
described here was staged, committed, stashed, reset, deleted, or moved.

## Git state

- Branch: `paper-pcr40-results`
- HEAD commit: `9350d341f8f720c95c181e9d1ddb9d3e58209f48` — **"Add PCR 40-step paper results"**
  (2026-08-19). **Important scope note:** this commit is from the legacy pre-v3 pipeline,
  not the final v3 system. `git show --stat HEAD` shows it added
  `whitebox_paper_results/pcr_full_completion_v1/` plus `build_global_eligible_manifest.py`,
  `generate_paper_rerun_sbatch.py`, `generate_sample_manifest.py`, `run_clean_baseline.py`,
  and `final_wga_*`/`final_wga_cueonly_*`/`final_wga_factonly_*` result files — the same
  WGA/FactOnly/CueOnly lineage already moved to `whitebox_attack/archive/pre_v3_root_20260910/`
  in an earlier cleanup pass. The branch name should not be read as describing the v3
  system's PCR results.
- `git status --porcelain` (85 lines total): 7 modified tracked files
  (`.gitignore`, `README.md`, `open_deep_research/src/open_deep_research/{configuration,deep_researcher,prompts,state,utils}.py`,
  `whitebox_attack/README.md`, `attack/metrics.py`), 44 renames (legacy
  `whitebox_attack/*.py` and `whitebox_attack/sbatch/*.sbatch` moved into
  `whitebox_attack/archive/pre_v3_root_20260910/`, some also modified), 25 deletions
  under `whitebox_results/` (superseded figures/JSONL from the pre-v3 pipeline), and
  4 untracked paths: `PUBLISHING.md`, `chapter3_methodology_audit/`, `attack/`,
  `whitebox_paper_results/e2e_full_system_v3/`.
- Untracked v3 code and results: the entire `attack/` directory (all `.py`
  files, including the pre-existing `analysis/` package) and the
  entire `whitebox_paper_results/e2e_full_system_v3/` results tree have never been
  committed to git — there is no git-based tamper-evidence for either.
- Modified ODR files (diff vs. `HEAD`, captured verbatim in `pre_cleanup_git_diff.patch`):
  `configuration.py` (+58/-? see patch), `deep_researcher.py` (682 insertions / 98
  deletions), `prompts.py`, `state.py`, `utils.py` — 762 insertions / 106 deletions total
  across the 5 files.
- `attack/metrics.py` also modified vs. `HEAD`: 42 insertions / 25 deletions.

## Relevant untracked files (full list captured in `pre_cleanup_sha256.txt`)

- `attack/*.py` — 27 top-level modules, including the 16 files already
  identified as under the "Active-Job Lock" in `PUBLISHING.md`, plus 11 more
  (`cue_development_report.py`, `dev_validation_data.py`, `odr_smoke.py`, `qwen_gate.py`,
  `retriever_calibration.py`, `answer_evaluator_v2.py`, `evaluate_completed_shard.py`,
  `final_production_evaluation.py`, `generate_pcr_cdf_tables.py`, `hop_aware_evaluation.py`,
  `musique_scope_audit.py`, `qwen14b_architecture_mechanism_audit.py`,
  `rescore_completed_answer_v2.py`, `__init__.py`) that are not on the production
  import-lock list but are still part of the untracked v3 tree.
- `analysis/` — the pre-existing, already-tested analysis package
  (18 files: `answer_evaluation.py`, `cli.py`, `completion_inventory.py`, `config.py`,
  `evidence.py`, `pairing.py`, `pcr.py`, `repr_parsing.py`, `statistics.py`,
  `trace_parser.py`, `trace_schema.py`, `__init__.py`, 5 tests under `tests/`, and the
  `final_paper_evaluation/` subpackage — `answer_harm.py`, `figures.py`, `master.py`,
  `mechanism.py`, `paper_tables.py`, `recovery.py`, `termination_routes.py`, `__init__.py`).
  **This package is treated as the "latest v3 result-producing implementation" for
  Phase 2/6 of this task** (see `legacy_metric_inventory.md` for the justification) —
  none of its files are modified or moved by this task; they are read-only inputs.

## Active/queued production jobs at time of snapshot

```
147621_[90-107%1]  v3_multihoprag_gemma27b     PENDING (JobArrayTaskLimit)
147621_89          v3_multihoprag_gemma27b     RUNNING (ml01)
149855_[0-107%1]   v3_hotpot_gpt_oss_20b       PENDING (DependencyNeverSatisfied)
149856_[0-107%1]   v3_2wiki_gpt_oss_20b        PENDING (DependencyNeverSatisfied)
149857_[0-107%1]   v3_musique_gpt_oss_20b      PENDING (DependencyNeverSatisfied)
149858_[0-107%1]   v3_multihoprag_gpt_oss_20b  PENDING (DependencyNeverSatisfied)
```

**Consequence for this task:** `gemma27b`/multihoprag is actively running and
backfilling; `gpt_oss_20b`'s four arrays are pending (blocked on a permanently-failed
dependency, see `PUBLISHING.md`). Per the task's Phase 0 instruction, every file on the
production import-lock list (`production_models.py`, `production_manifest.py`,
`cues.py`, `benchmark_search.py`, `corpus.py`, `evidence.py`, `hashing.py`,
`trace_schema.py`, `runner.py`, `runtime_gate.py`, `production_runner.py`, and the 5
modified `open_deep_research` core files) is treated as **strictly read-only** for the
remainder of this task. Every new file created lives under
`analysis/canonical/`.

## Model/dataset/condition completion status at time of snapshot

| Model | Shard files on disk | Status |
|---|---|---|
| `qwen14b` | 432 / 432 | Complete |
| `llama8b` | 432 / 432 | Complete |
| `gemma27b` | 414 / 432 | Partial — multihoprag actively backfilling (89→90+ of 108 as of this snapshot) |
| `gpt_oss_20b` | 0 / 432 | Blocked — see `PUBLISHING.md` (`v3_recovery_gate_gpt_oss_20b`, job 149854, failed 2026-09-08; all 4 arrays carry `afterok:149854_*`, permanently unsatisfiable) |
| `qwen7b` | n/a (validation-only) | Quarantined at `whitebox_paper_results/e2e_full_system_v3/production/quarantine/qwen7b_pipeline_validation_20260828_174947/` |

Full per-cell (model × dataset × condition) counts are computed fresh in
`final_sample_counts.csv` (Phase 9F) rather than restated here, since this inventory is a
point-in-time snapshot and the gemma27b/multihoprag count will keep changing while its
job runs.

# Attack: v3 Benchmark-Local ODR Production Experiment

This package contains the final production infrastructure for running Open
Deep Research on frozen multi-hop benchmark manifests with deterministic
local retrieval, plus the observation-manipulation attack under study.

Only the search provider is replaced. The ODR graph, prompts, supervisor,
researcher subgraph, `think_tool`, `ResearchComplete`, compression, and
final report generation all execute through `open_deep_research`.

There is only one version of this system -- "v3" was an internal development
label during earlier iteration and has been dropped from every path and
import; this is simply the final, current implementation.

## Layout

```text
attack/
├── __init__.py
├── metrics.py            # em_f1, mcnemar_test, paired_bootstrap_ci, wilson_ci -- shared by
│                          # runtime/production_runner.py and the analysis package
├── cue_definitions.py     # PERTURBATION_FAMILIES -- shared cue text, imported by runtime/cues.py
├── runtime/               # the frozen production system -- imported live by SLURM jobs;
│   │                      # do not edit while any job is queued/running (see below)
│   ├── production_runner.py    # SLURM array shard driver
│   ├── runner.py                # single-condition / paired-replay execution
│   ├── benchmark_search.py      # deterministic BM25 retrieval + one-shot cue injection
│   ├── runtime_gate.py           # pre-flight validation (embedded in each sbatch script)
│   ├── trace_schema.py            # production trace invariants (validate_branch_trace, etc.)
│   ├── hashing.py
│   ├── cues.py                     # 12-condition registry + verbatim suffix text
│   ├── corpus.py                    # per-dataset corpus/evidence row builders
│   ├── evidence.py                   # dependency-coverage computation
│   ├── production_manifest.py         # frozen 450-example manifest freeze
│   ├── production_models.py            # final model allowlist + parser map
│   └── musique_scope_audit.py           # MuSiQue stratified-sampling validation
├── launch/
│   └── generate_production_arrays.py   # matrix + sbatch + validation + submission generator
├── answer_scoring/        # the validated semantic-accuracy procedure -- current, live,
│   │                      # imported by ../analysis/answer_evaluation.py and
│   │                      # ../analysis/final_paper_evaluation/master.py
│   └── answer_evaluator_v2.py
└── tests/                 # unittest suite for everything above (70 tests)
```

Pre-canonical, superseded evaluators and one-off dev/calibration scripts
(never imported by any live sbatch-launched process or by `../analysis/`)
have been moved out of this package entirely, to `../archive/legacy_evaluators/`
and `../archive/dev_tools/` respectively -- kept on disk for provenance, not
git-tracked, not part of the publishable `attack/` tree. `../archive/legacy_evaluators/`
also holds `rescore_completed_answer_v2.py` and `evaluate_completed_shard.py`,
which `answer_scoring/answer_evaluator_v2.py` does **not** depend on (verified:
it has no imports from either).

The canonical analysis package lives at the repo-root sibling `../analysis/`
(not nested under `attack/`) -- see its own `README.md`.

## Final Matrix / Model Status (as of 2026-09-14)

| Model | Status |
|---|---|
| `qwen14b` | **Complete** -- 432/432 shards |
| `llama8b` | **Complete** -- 432/432 shards |
| `gemma27b` | **Complete** -- 432/432 shards (finished 2026-09-13) |
| `gpt_oss_20b` | **Blocked** -- 0/432 shards; all 4 production arrays carry a dead `afterok` dependency on a failed gate job (see repo-root `PUBLISHING.md`); being fixed and resubmitted |

Validation-only model: `qwen7b` (never part of the final production matrix;
its data is quarantined under `results/quarantine/`, not aggregated into any
final claim).

Final datasets: `hotpot`, `2wiki`, `musique`, `multihoprag`.

Final conditions: 12, defined in `runtime/cues.py` (see
`../analysis/canonical/condition_metadata.csv` for the full verbatim
cue-text table).

The generated final matrix is:

```text
results/matrix/v3_final_production_matrix.json
```

(This is the original, from the actual production launch. The two
2026-09-14 package reorganizations each produced their own re-validation
snapshot, preserved alongside under
`v3_final_production_matrix_POST_REORG_REGENERATION_20260914.json` and
`v3_final_production_matrix_POST_TOPLEVEL_REORG_REGENERATION_20260914.json`
-- proving neither reorganization changed production behavior; neither
overwrites the historical original, which is what `results/matrix/v3_final_production_matrix.json`
still is. Same pattern for `results/prelaunch/PRELAUNCH_REPORT.json`.)

## Commands

Generate or refresh matrix artifacts:

```bash
/home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m attack.launch.generate_production_arrays \
  --mode generate \
  --production-root results \
  --local-files-only
```

Validate before launch:

```bash
/home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m attack.launch.generate_production_arrays \
  --mode validate \
  --production-root results \
  --local-files-only
```

Submit missing final arrays without duplicating active jobs:

```bash
/home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m attack.launch.generate_production_arrays \
  --mode submit \
  --production-root results
```

Run the canonical analysis (PCR, route decomposition, evidence, answer
accuracy) -- see `../analysis/README.md` for full detail:

```bash
python -m analysis.cli --stage all \
  --production-root results \
  --output-root results/evaluation/v3_final_completed_qwen_llama_gemma_20260913
```

The historical, now-archived `final_production_evaluation.py` aggregate
command still runs but is **not** the canonical source for new claims:

```bash
/home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m archive.legacy_evaluators.final_production_evaluation \
  --production-root results
```

## Active-Job Lock

The current SLURM scripts (regenerated 2026-09-14 to reference the new
module paths, validated end-to-end via `--mode validate` and the full
test suite) import this package by module name from the working tree.
While final jobs are queued or running (currently: `gpt_oss_20b`'s four
arrays), keep these runtime files stable:

- `runtime/production_models.py`
- `runtime/production_manifest.py`
- `runtime/cues.py`
- `runtime/benchmark_search.py`
- `runtime/corpus.py`
- `runtime/evidence.py`
- `runtime/hashing.py`
- `runtime/trace_schema.py`
- `runtime/runner.py`
- `runtime/runtime_gate.py`
- `runtime/production_runner.py`
- `launch/generate_production_arrays.py`
- `metrics.py`, `cue_definitions.py`
- `open_deep_research/src/open_deep_research/*.py`

Renames, deletions, or behavioral cleanup for those files should wait until
the final jobs are finished, or the jobs should be cancelled and resubmitted
from the new code.

## Reorganization history

- 2026-09-13: `whitebox_attack/v3/`'s flat ~25-script directory split into
  `runtime/`, `launch/`, `legacy_evaluators/`, `dev_tools/` subpackages.
- 2026-09-14: entire repo restructured to drop the "whitebox_"/"v3"
  versioning: `whitebox_attack/v3/*` → `attack/*`,
  `whitebox_attack/v3/analysis/` → top-level `analysis/`,
  `whitebox_paper_results/e2e_full_system_v3/production/` → top-level
  `results/`, all legacy pre-v3 code and superseded result directories
  (`e2e_full_system_v1/v2`, `pcr_full_completion_v1`/`v2_60`,
  `e2e_final_answer_v1`) → `archive/`. See
  `../analysis/canonical/reorg_prep/` for the exact scripts that performed
  both moves and every rewritten import.
- 2026-09-14 (same day, second pass): `legacy_evaluators/` and `dev_tools/`
  removed from `attack/` entirely. The one piece of `legacy_evaluators/`
  actually still live-imported by `../analysis/` (`answer_evaluator_v2.py`,
  the validated semantic-accuracy procedure) moved to `attack/answer_scoring/`;
  everything else in both directories (self-tested only, not imported by
  `../analysis/` or any runtime file) moved to `../archive/legacy_evaluators/`
  and `../archive/dev_tools/`. `attack/` now contains only the current attack
  itself: `runtime/`, `launch/`, `answer_scoring/`, `tests/`, plus the two
  top-level shared modules (`metrics.py`, `cue_definitions.py`).

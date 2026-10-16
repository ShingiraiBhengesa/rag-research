# REPOSITORY_RESULTS_INVENTORY.md

**Read-only inventory.** No files were moved, renamed, deleted, regenerated,
or modified to produce this document, and no experiments were rerun. All
statements below were verified against the repository on disk at the time
of writing (2026-09-18), not recalled from memory.

> **This is a dated repository audit — a point-in-time snapshot from
> 2026-09-18**, taken at commit `9c327466ab8e730ba29a71db375ea5aa7556981d`
> with a 66-entry dirty working tree (21 modified, 40 deleted, 5
> untracked, per §1). It is preserved below **exactly as originally
> written**, as historical provenance of what the repository looked like
> at that point — every commit hash, file path, and
> "untracked"/"uncommitted" status claim in §1 through §9 describes that
> 2026-09-18 state only, not the repository's current state.
>
> **Current pushed branch tip** (`paper-pcr40-results`, as of this note):
> `dcc2efee8830b7c5ad17bc20c26f8ee703d7dc7a`. Local `HEAD` has since
> advanced one commit further, to `a0f701fda87e600872216cdbeaf04dcc561cc7a2`
> ("Remove superseded pre-canonical evaluation outputs"), not yet pushed
> at the time of this note. Neither hash was reachable from the
> 2026-09-18 snapshot commit above when this document was first written.
>
> **What has changed since the 2026-09-18 snapshot, resolving items this
> document flagged as open:**
> - §1/§5/§9(b)'s "deleted-but-uncommitted"
>   `results/evaluation/v3_final_completed_qwen_llama_gemma_20260913/`
>   and (per §1's dirty-tree count) `analysis/canonical/all_models_20260913/`
>   are both now **committed as deletions** (commit
>   "Remove superseded pre-canonical evaluation outputs"), confirmed
>   superseded by the canonical bundle below and referenced nowhere in
>   current code.
> - §1's untracked `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/` (the PCR-bundle
>   generator directory) and `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` (its output) are
>   now both **committed**. The generator directory being untracked was
>   itself a separately-discovered reproducibility gap (a clean checkout
>   could not have regenerated the committed bundle) — since fixed,
>   together with an explicit, locked model-scope constant
>   (`src/prematurestop/analysis/fixed_cue_matrix/results_bundle/scope.py`'s `CANONICAL_BUNDLE_MODELS`)
>   so the bundle no longer implicitly inherits its model scope from the
>   mutable `prematurestop.evaluation.config.INCLUDED_MODELS`.
> - Two files exceeded GitHub's plain-git size limits on push —
>   `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_run_level.csv` (~126 MB,
>   over the 100 MB hard limit) and
>   `results/whitebox/e1_fixed_cue_matrix/qwen_llama/qwen_llama_results_workbook.xlsx` (~54 MB,
>   over the 50 MB warning threshold). Both are now tracked via **Git
>   LFS** (`.gitattributes`); this required rewriting the (at the time,
>   still-unpushed) commits that introduced them, which is why the commit
>   hashes above post-date and do not match any hash this document or
>   earlier session history may have quoted before the migration.
> - §1's untracked `src/prematurestop/system/tool_parsers/gemma3_tool_code.py` and
>   `tests/system/test_gemma3_tool_code_parser.py` (the corrected
>   gemma27b vLLM tool-call parser and its test) were audited separately
>   for completion, test status, and whether committing them could be
>   mistaken as validating the still-quarantined gemma27b production
>   results; see that audit's own commit for the outcome, not restated
>   here.
>
> Original 2026-09-18 findings follow, unmodified:

---

## 1. Repository overview

| | |
|---|---|
| Repository root | `$HOME/rag-research` |
| Current branch | `paper-pcr40-results` |
| Current commit (`HEAD`) | `9c327466ab8e730ba29a71db375ea5aa7556981d` |
| Matches the frozen commit `9c327466...`? | **Yes** — `HEAD` is exactly that commit |
| Worktree clean? | **No.** `git status --porcelain` shows 66 entries: 21 modified, 40 deleted (all under one pre-existing historical evaluation directory, see §5), 5 untracked new paths. No tracked file's *content* was altered by the inventory task itself — these changes predate this task and were already present when it started. |

Untracked (new, not yet committed) top-level paths:
```
?? src/prematurestop/analysis/fixed_cue_matrix/results_bundle/
?? src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/
?? src/prematurestop/system/tool_parsers/gemma3_tool_code.py
?? tests/system/test_gemma3_tool_code_parser.py
?? results/whitebox/e1_fixed_cue_matrix/qwen_llama/
```

Deleted-but-uncommitted (present in `HEAD`'s tree, absent from the working
directory): all 40 files under
`results/evaluation/v3_final_completed_qwen_llama_gemma_20260913/` — the
directory itself no longer exists on disk. This predates the current
inventory task; not investigated further here beyond noting it (see §5).

**Important top-level directories:**

| Directory | Role |
|---|---|
| `attack/` | Attack implementation: runtime harness, launch/sbatch generation, answer scoring |
| `analysis/` | Post-hoc analysis code (reads traces, never modifies them) |
| `results/` | Production data: traces, runs, manifests, logs, and several dated one-off evaluation snapshots |
| `archive/` | Superseded code/results kept for provenance; `archive/legacy_results/` alone holds ~172,000 files (mostly per-trace `final_report.md`/similar), git-ignored |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` | Current canonical PCR + evidence(final-report-input) + answer results deliverable |
| `open_deep_research/` | Third-party ODR agent framework dependency (not ours, not inventoried) |
| `chapter3_methodology_audit/` | Small standalone audit, unrelated to the Qwen/Llama results consolidation |

---

## 2. Directory tree (results-relevant paths only)

Paths are relative to the repository root. Bulky/unrelated content (raw
per-trace JSON under `results/traces/`, `results/runs/`, `.pyc` files,
`archive/legacy_results/`'s ~172k per-trace reports) is summarized by count,
not enumerated.

```
results/
├── traces/{qwen14b,llama8b,gemma27b,gpt_oss_20b}/branches/{dataset}/{example_id}/{condition}/trace.json
│     qwen14b: 21,599 trace.json files on disk. llama8b: 21,074.
│     (96 cells x 450 = 43,200 expected per qwen+llama combined; the gap is
│     unrun/execution-failed shards, consistent with completion_inventory's
│     own valid-row count of 42,547 -- see §8.)
├── runs/{model}/...                      # per-shard raw result JSONL, one dir per model
├── manifests/{dataset}_n450.jsonl        # dataset annotations (question/gold answer/aliases)
├── evaluation/                           # historical snapshots, see §5 -- NOT current
│   ├── _archive_superseded_20260910/     # explicitly named "superseded"
│   ├── final_qwen_llama_paper_analysis_20260910/
│   └── v3_final_completed_qwen_llama_20260910/
└── logs/, sbatch/, submissions/, quarantine/, recovery_20260905/, recovery_20260914/

analysis/
├── canonical/
│   ├── canonical_completion.py           # THE canonical PCR source of truth
│   ├── chapter3_final_source_of_truth.md # locked definitions + the 2026-09-13 fix note
│   ├── test_canonical_completion.py
│   ├── retrieval_equivalence.py          # retrieval-equivalence fingerprinting (reused)
│   ├── reconcile.py
│   ├── cleanup_change_log.md, pre_cleanup_inventory.md, legacy_metric_inventory.md,
│   │   canonical_vs_previous_comparison.md, retrieval_equivalence_summary.md  # process/audit docs, see §5
│   ├── final_gap_audit/                  # gap-closure audit docs (gemma/direct-exit focused)
│   ├── gemma27b_analysis/                # gemma-specific (out of Qwen/Llama scope)
│   ├── reorg_prep/                       # one-off migration scripts from the Sept 14 restructure
│   └── results_bundle/                   # generates results/whitebox/e1_fixed_cue_matrix/qwen_llama/ -- see canonical source map
│       ├── build_bundle.py               # top-level CLI orchestrator
│       ├── comparisons.py                # the single reconciled comparison manifest
│       ├── pcr_full.py, paired_deltas.py, run_inventory.py
│       ├── evidence_ext.py, process_metrics.py, mechanism_ext.py
│       ├── answer_ext.py, retrieval_equivalence_ext.py
│       ├── gates.py, freeze_manifest.py
│       ├── figures_bundle.py, tables_bundle.py, workbook.py, docs_bundle.py
│       └── __init__.py
├── evidence_hop_coverage/                # generates the evidence/hop-coverage deliverable
│   ├── build_evidence_hop_coverage.py    # main script (run-level, summary, paired, gates, docs)
│   ├── hop_stratified_pcr.py             # PCR cut by hop/evidence-count stratum
│   ├── evidence_vs_answer_harm.py        # evidence-incompleteness vs answer-harm association check
│   ├── integrated_recovery_story.py      # recovery summary + evidence-boundary merge + gates 15-18
│   ├── README.md, EVIDENCE_HOP_RESULTS.md, EVIDENCE_VS_ANSWER_HARM_NOTES.md,
│   │   EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md
│   └── *.csv (see canonical source map)
├── final_paper_evaluation/               # older figures/tables/recovery logic, partially reused
│   ├── recovery.py                       # RECOVERY_* label constants, reused by process_metrics.py
│   └── figures.py, paper_tables.py, mechanism.py, master.py, answer_harm.py, ...
├── evidence.py                           # final-report-input + injection + episode-0 coverage boundaries
├── answer_evaluation.py                  # answer records, summaries, paired transitions
├── completion_inventory.py               # run inventory, `_row_is_valid`
├── trace_parser.py, trace_schema.py      # trace/event parsing (shared by everything above)
├── config.py                             # INCLUDED_MODELS, DATASETS, CONDITIONS, DETERMINISTIC_SEED
├── pcr.py                                # DEPRECATED legacy dual-denominator PCR -- see §5
└── cli.py                                # legacy `--stage pcr/statistics` pipeline, still wires src/prematurestop/evaluation/pcr.py

attack/
├── runtime/
│   ├── evidence.py                       # EvidenceDependency/EvidenceCoverage, live coverage definitions
│   ├── corpus.py                         # per-dataset EvidenceRow/EvidenceDependency builders (source of `required_paragraph_ids`/`hop_count`)
│   └── ... (runner, production_runner, runtime_gate, trace_schema, etc.)
├── answer_scoring/
│   └── answer_evaluator_v2.py            # semantic_correct / semantic_label / needs_adjudication logic
└── launch/, tests/

results/whitebox/e1_fixed_cue_matrix/qwen_llama/                 # CURRENT canonical PCR bundle
├── README.md, MASTER_RESULTS.md, CHAPTER5_WRITING_NOTES.md, LAB_MEETING_BRIEF.md
├── analysis_manifest.json
├── data/  (16 CSVs -- 13 required + mechanism_run_level/summary + VALIDATION_GATES)
├── tables/  (6 .md files)
├── appendix/  (4 .tex files)
├── figures/  (432 files across 6 subdirectories)
└── qwen_llama_results_workbook.xlsx

archive/
├── legacy_evaluators/                    # pre-rename PCR/evaluation code (e.g. "premature completion")
├── legacy_results/                        # ~172,000 files, per-trace reports from pre-restructure runs
├── code/, dev_tools/
```

---

## 3. Canonical source map

| Result family | Canonical input | Generating script | Canonical output | Status |
|---|---|---|---|---|
| Run inventory & eligibility | `results/traces/{qwen14b,llama8b}/**`, `results/runs/**`, `results/manifests/**` | `src/prematurestop/evaluation/completion_inventory.py::build_inventory` → `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/run_inventory.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/run_inventory.csv` | CURRENT |
| PCR stepwise values | Canonical completion records | `src/prematurestop/evaluation/completion.py` → `results_bundle/pcr_full.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/{pcr_run_level,pcr_stepwise}.csv` | CURRENT |
| Paired PCR deltas | Canonical completion records + `comparisons.py` | `results_bundle/paired_deltas.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/pcr_paired_deltas_stepwise.csv` | CURRENT |
| Route decomposition | Canonical completion records | `results_bundle/pcr_full.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/route_decomposition_stepwise.csv` | CURRENT |
| Process/tool metrics | Canonical + evidence records | `results_bundle/process_metrics.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/process_metrics.csv` | CURRENT |
| Local-completion evidence coverage | Traces (re-parsed) | `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/build_evidence_hop_coverage.py` | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/run_level_evidence_hops.csv` (`local_coverage_fraction`), `evidence_hop_summary.csv` | CURRENT |
| Hop/evidence-count strata | Traces (`hop_count`/`required_paragraph_ids`) | `build_evidence_hop_coverage.py`; PCR cut: `hop_stratified_pcr.py` | `run_level_evidence_hops.csv` (`dataset_stratum_value`); `hop_stratified_pcr_stepwise.csv`, `hop_stratified_pcr_paired_deltas.csv` | CURRENT |
| Paired local evidence loss | Evidence-hop run-level + `comparisons.py` | `build_evidence_hop_coverage.py` | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/paired_evidence_deltas.csv` | CURRENT |
| Final-report-input evidence coverage | Traces (citation parsing) | `src/prematurestop/evaluation/evidence.py::build_evidence_records` → `results_bundle/evidence_ext.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/{evidence_run_level,evidence_summary}.csv` | CURRENT |
| Answer run-level evaluation | Traces + manifests + `answer_evaluator_v2.py` | `src/prematurestop/evaluation/answer_evaluation.py` → `results_bundle/answer_ext.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_run_level.csv` | CURRENT |
| Answer summaries | Answer records | `answer_evaluation.py::summarize_by_cell` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_summary.csv` | CURRENT (pre-adjudication) |
| Paired answer flips | Answer records + `comparisons.py` | `answer_evaluation.py::compute_paired_answer_transitions` → `answer_ext.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_paired_flips.csv` | CURRENT (pre-adjudication) |
| Evidence-incompleteness vs answer harm | `run_level_evidence_hops.csv` + answer records | `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/evidence_vs_answer_harm.py` | `evidence_incomplete_vs_answer_harm.csv`, `EVIDENCE_VS_ANSWER_HARM_NOTES.md` | CURRENT, explicitly preliminary/association-only |
| Retrieval-equivalence sensitivity | Traces (search fingerprints) + `comparisons.py` | `src/prematurestop/analysis/fixed_cue_matrix/audits/retrieval_equivalence.py` → `results_bundle/retrieval_equivalence_ext.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/retrieval_equivalence.csv` | CURRENT |
| Validation gates (PCR bundle) | All `data/*.csv` above | `results_bundle/gates.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/VALIDATION_GATES.csv` | CURRENT — 12/13 PASS (see §8) |
| Validation gates (evidence/hop) | `run_level_evidence_hops.csv` etc. | `build_evidence_hop_coverage.py` (gates 1-14) + `integrated_recovery_story.py` (gates 15-18) | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/validation_gates.csv` | CURRENT — 18/18 PASS |
| Supervisor recovery summary | `process_metrics.csv::later_evidence_recovery` | `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/integrated_recovery_story.py` | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/supervisor_recovery_summary.csv` | CURRENT |
| Evidence-boundary merge | `evidence_summary.csv` + `paired_evidence_deltas.csv` | `integrated_recovery_story.py` | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/evidence_boundary_paired_results.csv` | CURRENT |
| Figures | `data/*.csv` | `results_bundle/figures_bundle.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/figures/**` (432 files) | CURRENT |
| Appendix tables (LaTeX) | `data/*.csv` | `results_bundle/tables_bundle.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/appendix/*.tex` | CURRENT |
| `.xlsx` workbook | `data/*.csv` | `results_bundle/workbook.py` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/qwen_llama_results_workbook.xlsx` | CURRENT |
| Legacy PCR (dual-denominator) | Traces | `src/prematurestop/evaluation/pcr.py`, wired via `src/prematurestop/analysis/fixed_cue_matrix/cli.py --stage pcr` | No canonical bundle output; scattered legacy dirs under `results/evaluation/` | **DEPRECATED — do not use for new work.** Coexists in the codebase with the canonical implementation; a reviewer unfamiliar with the history could mistakenly import this one. |

---

## 4. Markdown report map

| Path | Purpose | Data boundary | Canonical/current? | Overlaps with | Recommended disposition |
|---|---|---|---|---|---|
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` | Main PCR+evidence+answer narrative report | PCR/control-flow + final-report-input evidence + final answer | Yes | `CHAPTER5_WRITING_NOTES.md` (same tables embedded) | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/CHAPTER5_WRITING_NOTES.md` | Chapter-5 writing aid, requirement→file map + embedded tables | Same as above | Yes | `MASTER_RESULTS.md` | KEEP AS SUPPORTING |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/LAB_MEETING_BRIEF.md` | 6-slide compact brief | PCR + evidence + answer, compact | Yes | `MASTER_RESULTS.md` (subset) | KEEP AS SUPPORTING |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/README.md` | Bundle regeneration/methodology doc | N/A (meta) | Yes | — | KEEP AS SUPPORTING |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/TABLE_INDEX.md` | Sign-convention + evidence-boundary disambiguation | Cross-cutting | Yes | — | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/evidence_tables.md` | Paired evidence-loss table | Final-report-generation input | Yes | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/paired_evidence_deltas.csv` (different boundary, documented) | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/answer_tables.md` | Per-cell answer summary + paired flips | Final generated answer | Yes | — | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/full_pcr_tables.md`, `full_route_tables.md`, `main_chapter_tables.md` | Full/compact PCR and route tables | PCR/control-flow | Yes | `data/pcr_stepwise.csv` etc. (formatted view) | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/README.md` | Methodology + boundary reconciliation | Cross-cutting | Yes | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/TABLE_INDEX.md` | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/EVIDENCE_HOP_RESULTS.md` | Evidence/hop-coverage narrative | Researcher local completion at T_i | Yes | `MASTER_RESULTS.md` §"Evidence coverage" (different boundary) | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/EVIDENCE_VS_ANSWER_HARM_NOTES.md` | Association check, evidence loss vs answer harm | Local completion + final answer | Yes, explicitly preliminary | `EVIDENCE_HOP_RESULTS.md`, `MASTER_RESULTS.md` answer tables | KEEP AS SUPPORTING |
| `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md` | Integration report (recovery, both evidence boundaries, hop-stratified PCR) | Cross-cutting | Yes | All of the above (explicitly, by design — it consolidates them) | KEEP AS SUPPORTING |
| `results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md` | Locked PCR definitions + the 2026-09-13 fix history | PCR/control-flow | Yes (referenced by `MASTER_RESULTS.md`) | — | KEEP AS CANONICAL |
| `results/whitebox/e1_fixed_cue_matrix/canonical/cleanup_change_log.md`, `pre_cleanup_inventory.md`, `legacy_metric_inventory.md`, `canonical_vs_previous_comparison.md` | Repo-restructure process docs | N/A (meta/process) | Historical, not results | — | ARCHIVE CANDIDATE |
| `results/whitebox/e1_fixed_cue_matrix/canonical/retrieval_equivalence_summary.md` | Ad-hoc retrieval-equivalence audit, all 4 models pooled, dated 2026-09-11 | PCR/control-flow-adjacent (retrieval) | **No** — predates the qwen/llama-only scope lock and the current per-comparison `retrieval_equivalence.csv`; different denominator (64,800 pairs across all 4 models, not the current comparison-manifest-scoped counts) | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/retrieval_equivalence.csv` | ARCHIVE CANDIDATE |
| `src/prematurestop/analysis/fixed_cue_matrix/audits/final_gap_audit/*.md` (7 files) | Gemma/direct-exit gap-closure audits | Mixed (mostly PCR/control-flow, gemma-focused) | Not Qwen/Llama-scoped; useful precedent for blinded-review methodology (`direct_exit_adjudication_guide.md`) | — | KEEP AS SUPPORTING (methodology precedent only) |
| `results/whitebox/e1_fixed_cue_matrix/canonical/gemma27b_results_synthesis.md` | Gemma27b-specific synthesis | N/A — different model | Out of scope for this inventory's Qwen/Llama focus | — | KEEP AS SUPPORTING (different model) |
| `results/evaluation/_archive_superseded_20260910/**` (~40 `.md` files) | Historical PCR/answer/mechanism-audit reports, dated Sept 3-10 | PCR/control-flow + final answer | **No** — directory is explicitly named "superseded"; predates the 2026-09-13 forced-search branch-point fix | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/*` | ARCHIVE CANDIDATE |
| `results/evaluation/final_qwen_llama_paper_analysis_20260910/{README,REPORT,validation_report}.md` | An earlier "final" paper-analysis snapshot, dated Sept 10-11 | PCR/control-flow + final answer | **No** — predates the fix; superseded by `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/*` | ARCHIVE CANDIDATE |
| `results/evaluation/v3_final_completed_qwen_llama_20260910/**` (12 `.md` files) | Per-model/per-dataset PCR reports, dated Sept 10 | PCR/control-flow | **No** — confirmed stale (see §5 spot-check: PCR@1 26.1% here vs 25.9% in current canonical, different denominators) | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/full_pcr_tables.md` | ARCHIVE CANDIDATE |
| `docs/notes/PUBLISHING_legacy.md` (repo root) | Publishing checklist | N/A (ops/process) | **Stale** — describes gpt_oss_20b as "permanently stuck"/"unresolved," which this session's later work resolved (out of this inventory's Qwen/Llama scope to verify further) | — | UNKNOWN — NEEDS REVIEW |
| `README.md` (repo root) | Top-level repo README | N/A (meta) | Presumed current (not diffed against actual structure in this pass) | — | UNKNOWN — NEEDS REVIEW |

---

## 5. Duplicate and stale-output audit

**Filename-pattern search** (`*copy*`, `*(1)*`, `*_old*`, `*_v1*`, `*_v2*`, `*backup*`, `*_new*`, `*_final_final*`), excluding `archive/legacy_results/` and raw trace data: no results-table duplicates found. The only hits are legitimate current filenames (`src/prematurestop/evaluation/answer_scoring.py`, the actual current evaluator — not a stale copy) and already-archived legacy files (`archive/legacy_evaluators/rescore_completed_answer_v2.py`, `archive/code/results/*_v2.jsonl`).

**Repeated report names across directories** — the same relative filenames recur across three historical `results/evaluation/` snapshots and the current bundle:

| Filename pattern | Appears in | Relationship |
|---|---|---|
| `{model}_{dataset}_pcr.md` | `_archive_superseded_20260910/pcr_termination_completed_20260904/`, `.../pcr_termination_llama_completed_20260908/`, `.../pcr_termination_qwen_completed_20260906/`, `v3_final_completed_qwen_llama_20260910/`, plus current `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/full_pcr_tables.md` | **Different**, not identical or partially-overlapping copies — each is a full recomputation at a different point in the pipeline's history. Spot-check: `v3_final_completed_qwen_llama_20260910/by_model_dataset/qwen14b_hotpot_pcr.md` reports Neutral n=450, PCR@1=26.1%; current `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/main_chapter_tables.md` reports Neutral n=444 (eligible), PCR@1=25.9%. Both denominator and value differ — confirms staleness, not mere duplication. |
| `README.md` / `REPORT.md` / `validation_report.md` | Each of the three `results/evaluation/*` dated dirs, plus `results/whitebox/e1_fixed_cue_matrix/qwen_llama/README.md` | Different content per directory (each dated snapshot has its own); not byte-identical, not compared pairwise beyond the spot-check above (would require per-pair hashing not performed here — see note below). |
| `adjudication_packet.csv` | `results/evaluation/_archive_superseded_20260910/answer_evaluator_v2/` and `.../llama_completed_process_audit_20260908/` | Two files, same name, different parent dirs — **not compared byte-for-byte** in this pass; both predate the current `answer_run_level.csv` and its `needs_adjudication_flag` column. Schema differs from the current answer table (has `condition_blind_short_answer`, `human_label`, `human_notes` columns the current one lacks — see §6). |

No systematic pairwise hashing (`sha256sum`) was run across all ~150 relevant Markdown files or across the three historical evaluation directories in full — that would require materially more time than this inventory pass; the spot-check above establishes staleness by content inspection, sufficient to classify the three `results/evaluation/*` directories as pre-fix and superseded, but a reviewer wanting a complete pairwise diff should treat that as separate follow-up work, not assumed-complete here.

**Deleted-but-uncommitted directory**: `results/evaluation/v3_final_completed_qwen_llama_gemma_20260913/` — present in the `HEAD` commit's tree (40 files: PCR CSVs/MDs for qwen14b/llama8b/gemma27b across all 4 datasets, plus an answer summary and a paired-checkpoint-stats file), **absent from the actual working directory on disk**. This predates the current inventory task. Cannot be hash-compared to anything since it no longer exists on disk; its git-tracked content could still be recovered via `git show HEAD:<path>` if needed, but that is a decision for the consolidation task, not this one.

**Counts/commit divergence from frozen results**: confirmed for all three `results/evaluation/*` historical directories (dated 2026-09-03 through 2026-09-11, i.e. before the 2026-09-13 forced-search branch-point fix documented in `results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md` line 646 — "Correction (2026-09-13): forced-search branch-point fix changes llama8b's numbers too"). All current canonical numbers in `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` and `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/` were generated at `HEAD` (`9c327466...`), after that fix.

---

## 6. Semantic-accuracy status

**Located:**
- `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_run_level.csv` — present. Columns: `model, dataset, condition, example_id, gold_answer, gold_aliases, raw_final_report, extracted_short_answer, extraction_reason, automatic_semantic_label, exact_match, token_f1, gold_containment_diagnostic_only, needs_adjudication_flag`.
- `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_summary.csv` — present (per-cell aggregates: `n_primary_denominator`, `pct_semantic_correct`, `exact_match_mean`, `token_f1_mean`, `n_needs_adjudication`, etc.).
- `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_paired_flips.csv` — present (`both_correct`, `control_correct_cue_wrong`, `control_wrong_cue_correct`, `both_wrong`, `net_harm`, `net_harm_pct_points`).

**Implementation that creates `semantic_correct`**: `src/prematurestop/evaluation/answer_scoring.py::semantic_accuracy()`. Returns a `SemanticResult(label, correct_bool, reason, matched)` tuple; `correct_bool` is `True` only for `label == "correct"`. Decision cascade (in order): official exact-match → yes/no semantic match → list-answer match → per-choice negation/semantic-equivalence/phrase-containment check → token-F1 ≥ 0.85 fallback → default `"incorrect"`.

**Implementation and rule for `needs_adjudication_flag`**: set in `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/answer_ext.py::write_answer_run_level_csv` (line ~71) as `r.semantic_label == "needs_adjudication"`. The underlying `"needs_adjudication"` label itself is assigned by `semantic_accuracy()` in three cases: a genuinely conflicting/ambiguous extracted answer, a multi-candidate answer that contains the gold answer among other candidates, or high token-F1 (≥0.85) without a "safe" (exact/semantic/phrase) match.

**Flagged-row counts** (from `answer_run_level.csv`, current canonical):

| Model | Dataset | Flagged / Total |
|---|---|---|
| llama8b | 2wiki | 56 / 5,318 |
| llama8b | hotpot | 107 / 5,262 |
| llama8b | multihoprag | 93 / 5,183 |
| llama8b | musique | 91 / 5,208 |
| qwen14b | 2wiki | 41 / 5,397 |
| qwen14b | hotpot | 38 / 5,396 |
| qwen14b | multihoprag | 38 / 5,399 |
| qwen14b | musique | 79 / 5,384 |

Total: 543 flagged rows out of 42,547 valid answer rows (~1.3%).

**Existing adjudication queue or completed adjudication**: **none exists for the current canonical `answer_run_level.csv`.** Two OLD adjudication packets exist, but only under historical, superseded directories: `results/evaluation/_archive_superseded_20260910/answer_evaluator_v2/adjudication_packet.csv` and `.../llama_completed_process_audit_20260908/adjudication_packet.csv`. Neither has been regenerated against the current 543-row flagged set. A related but distinct precedent exists for a *different* adjudication question (whether a `NO_TOOL_DIRECT_EXIT` classification is correct, not whether an answer is semantically correct): `src/prematurestop/analysis/fixed_cue_matrix/audits/final_gap_audit/direct_exit_adjudication_guide.md` + its accompanying sample — this is a completed methodology for a different variable, reusable as a pattern but not itself an answer-accuracy adjudication.

**Columns available for manual review**: `answer_run_level.csv` already has `gold_answer`, `gold_aliases`, `raw_final_report` (contains the full generated report the question+predicted answer must be read from), `extracted_short_answer` (predicted answer), `automatic_semantic_label` (automatic label), and `needs_adjudication_flag` (flag reason is implicit in the label, not a separate free-text reason column). It does **not** currently have a standalone `question` column (the question text lives in `results/manifests/{dataset}_n450.jsonl`, joinable by `example_id`) or a free-text "flag reason" column distinct from the label itself.

**Whether condition labels can be hidden for blinded review**: `condition` is a plain column in `answer_run_level.csv` — not blinded by default. Two precedents exist for how blinding has been done before in this repo: (1) the old `adjudication_packet.csv` schema had a `condition_blind_short_answer` column plus an anonymized `review_id` instead of exposing `condition` directly; (2) `direct_exit_adjudication_guide.md`'s sample prefixes de-anonymization columns with `_` so they can be dropped/hidden during review and revealed afterward. Neither mechanism has been applied to the current 543-row flagged set — a blinded review packet would need to be generated fresh, following one of these two existing patterns.

**What is genuinely missing before semantic accuracy can be called final**:
1. A blinded (or at least explicitly reviewable) adjudication packet for the 543 currently-flagged rows, generated from current data (the two existing packets are stale and cover a different, older row set).
2. The actual human review pass itself — not started for the current canonical answer set.
3. A defined process for folding adjudicated labels back into `answer_summary.csv`/`answer_paired_flips.csv`'s aggregates (no such re-aggregation script currently exists).

**To be explicit, per the task's own instruction**: automatic semantic accuracy is **not missing** — it is fully computed, for every valid row, via a deterministic, documented rule (`answer_evaluator_v2.py::semantic_accuracy`). What is missing is specifically the *human-adjudicated* version for the 543 rows the automatic rule itself flagged as ambiguous, and the "blind" presentation layer for that review.

---

## 7. Missing-artifact audit

Every path referenced (backtick-quoted, ending in `.csv`/`.md`/`.tex`/`.png`/`.json`) inside the six named files was checked for existence relative to that file's own directory.

| Source file | Referenced paths checked | Result |
|---|---|---|
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` | 22 paths (13 `data/*.csv`, `tables/TABLE_INDEX.md`, 9 figure `.png` files, `chapter3_final_source_of_truth.md`) | **All PRESENT** |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/CHAPTER5_WRITING_NOTES.md` | 11 paths (4 `appendix/*.tex`, `data/*.csv` x3, `MASTER_RESULTS.md`, `tables/*.md` x2, a figures subdirectory) | **All PRESENT** |
| `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/EVIDENCE_HOP_RESULTS.md` | 6 paths (5 same-directory files + cross-package `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_summary.csv`) | **All PRESENT** |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/evidence_tables.md` | 1 path (`TABLE_INDEX.md`) | **PRESENT** |
| `results/whitebox/e1_fixed_cue_matrix/qwen_llama/tables/answer_tables.md` | 0 explicit file-path references | N/A |
| `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/EVIDENCE_VS_ANSWER_HARM_NOTES.md` | 3 paths (all same-directory) | **All PRESENT** |

**No MISSING, PRESENT BUT STALE, or PRESENT WITH UNCERTAIN PROVENANCE paths were found among these six files' own references.** All internal cross-references in the current canonical documents resolve correctly. (This does not extend to the historical `results/evaluation/*` documents, which were not part of the six named files and were not path-checked in this section — see §5 for their status instead.)

---

## 8. Validation status

**Frozen commit check**: `HEAD` = `9c327466ab8e730ba29a71db375ea5aa7556981d`, exactly the required frozen commit. Confirmed via `git rev-parse HEAD`.

### `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/VALIDATION_GATES.csv` — 12/13 PASS

| Gate | Description | Status |
|---|---|---|
| 1 | Qwen14B/Llama8B have the expected 12-condition, 4-dataset matrix | PASS (96/96 cells) |
| 2 | Every PCR curve is monotonic nondecreasing | PASS |
| 3 | No qualifying completion step exceeds 29 | PASS |
| 4 | pcr == research_complete_pcr + no_tool_exit_pcr on every row | PASS |
| 5 | Every paired N ≤ either condition's eligible N | PASS |
| 6 | No row silently imputed from another condition | PASS |
| 7 | Coverage/recall/PCR/accuracy values in [0,1] | PASS |
| 8 | Full-hop status agrees with complete evidence coverage | PASS |
| 9 | The four answer-flip cells sum to paired answer N | PASS |
| 10 | Every figure value joins exactly to a canonical CSV row | PASS |
| **11** | **Every number in MASTER_RESULTS.md has a table/file reference** | **FAIL — 17 lines with an un-cited bare number** (spot-check regex; the gate's own docstring calls this "review manually," not a hard failure; the flagged lines are the `_TODO(human)` narrative placeholders in sections 7.4-7.6, which genuinely have not been written yet) |
| 12 | Adjudication flag present, never silently mixed into "accuracy" | PASS |
| 13 | No legacy (WGAR/Attack-A/Cue-Only/Bridge-Fact) material in bundle CSVs | PASS |

### `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/validation_gates.csv` — 18/18 PASS

Gates 1-14 (unique IDs, counts match PCR, completion step/route match PCR, no completion value on MAX_BUDGET_EXIT, coverage/missing-count ranges, full-evidence-iff-zero-missing, local-excludes-post-completion, final≥local, provenance, paired-ID-identity, strata-agree-with-annotations, Oracle-Complete-anchor-coverage, manual-sample-coverage) all PASS. Gates 15-18 (hop-stratified PCR monotonic, hop-stratified paired N ≤ unstratified, recovery counts sum correctly, evidence-boundary merge complete) all PASS.

**No failures were silently repaired.** Gate 11's failure is reported as-is above; it was not modified, worked around, or hidden to produce this inventory.

---

## 9. Consolidation blockers

### (a) Blockers to writing Chapter 5 now
- `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` sections 7.4 (model comparison), 7.5 (dataset comparison), and 7.6 (main/negative findings) are still `_TODO(human)` placeholders — this is gate 11's actual failure. This is a genuine blocker: the numbers exist, the prose does not.
- Two evidence "coverage" numbers (final-report-input vs local-completion) coexist for the same comparisons and are *not* the same value (by design, per §4's boundary column) — a Chapter 5 draft must pick, per claim, which boundary it is citing, or it will silently mix two different measurements under one label. This is documented (`TABLE_INDEX.md`, `evidence_hop_coverage/README.md`) but not yet resolved into a single narrative choice.

### (b) Blockers to producing the reproducible final results package
- Three historical `results/evaluation/*` directories and `results/whitebox/e1_fixed_cue_matrix/canonical/retrieval_equivalence_summary.md` remain on disk alongside the current canonical outputs with no disposition decision recorded (§4, §5) — a "final" package needs an explicit decision (archive/keep-as-supporting) for each, not just an implicit "newest wins."
- `src/prematurestop/evaluation/pcr.py` (the deprecated legacy PCR implementation) is still importable and still wired into `src/prematurestop/analysis/fixed_cue_matrix/cli.py`. Nothing prevents a future contributor from accidentally using it instead of `canonical_completion.py`.
- The deleted-but-uncommitted `results/evaluation/v3_final_completed_qwen_llama_gemma_20260913/` directory (§1, §5) needs a decision (commit the deletion, or restore) before a "clean" package state can be declared.

### (c) Blockers to reporting final adjudicated semantic accuracy
- No blinded adjudication packet exists yet for the current 543 flagged rows (§6). Generating one is small, well-precedented work (two prior packet formats exist to model it on), but it has not been done.
- The actual human review pass has not been performed.
- No script exists to fold adjudicated labels back into `answer_summary.csv`/`answer_paired_flips.csv` — this would need to be built, however small, before "adjudicated" numbers could replace "automatic" ones in any report.

---

## Files to provide to the external reviewer

The smallest set needed to design the consolidation, without needing to re-derive anything already established in this inventory:

- `REPOSITORY_RESULTS_INVENTORY.md` (this file)
- `results/whitebox/e1_fixed_cue_matrix/qwen_llama/` (whole directory — the current canonical PCR/evidence/answer bundle: `MASTER_RESULTS.md`, `CHAPTER5_WRITING_NOTES.md`, `data/`, `tables/`, `appendix/`, `figures/`, `analysis_manifest.json`)
- `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage/` (whole directory — the current canonical evidence/hop/recovery bundle, including the four generator scripts)
- `results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md` (locked PCR definitions + the 2026-09-13 fix history that explains why pre-fix historical dirs differ)
- `src/prematurestop/evaluation/completion.py` and `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/` (the generating code for the PCR bundle, for anyone who needs to regenerate or extend it)
- `src/prematurestop/evaluation/evidence.py`, `src/prematurestop/evaluation/answer_evaluation.py`, `src/prematurestop/evaluation/answer_scoring.py`, `attack/runtime/{corpus,evidence}.py` (the shared building blocks both bundles depend on)
- This repository's `git log` for commit `9c327466ab8e730ba29a71db375ea5aa7556981d` and its immediate predecessors, for lineage confirmation

# Legacy metric inventory (Phase 2)

Purpose: identify every existing PCR/premature-completion-adjacent implementation
in the repository, classify each against the canonical definition adopted in
this package, and determine — with evidence, not filename assumption — which
implementation actually produced "the latest PCR tables and plots" prior to
this canonical package.

## Which implementation is "the latest"?

**Not the HEAD-commit "PCR 40-step" results.** `git show --stat HEAD` (commit
`9350d341`, "Add PCR 40-step paper results", on branch `paper-pcr40-results`)
shows it added `whitebox_paper_results/pcr_full_completion_v1/` plus
`build_global_eligible_manifest.py`, `generate_paper_rerun_sbatch.py`,
`generate_sample_manifest.py`, `run_clean_baseline.py`, and
`final_wga_*`/`final_wga_cueonly_*`/`final_wga_factonly_*` result files — the
same pre-v3 WGA/FactOnly/CueOnly lineage already moved to
`whitebox_attack/archive/pre_v3_root_20260910/` in an earlier cleanup. Per
task constraint 4 ("Do not use the older FactOnly, Bridge Fact, Cue Only,
WGA... designs unless you need to identify and explicitly exclude them"),
this commit's results are explicitly excluded here — the branch name must not
be read as describing the v3 system's PCR results.

**Determination: the latest v3 result-producing implementation is
`analysis/pcr.py`** (plus its dependencies
`trace_parser.py`, `trace_schema.py`, `completion_inventory.py`, `config.py`,
all under `analysis/`), whose outputs are the
already-generated CSVs at
`whitebox_paper_results/e2e_full_system_v3/production/evaluation/v3_final_completed_qwen_llama_20260910/pcr/`
(file mtimes 2026-09-10 00:25, the most recent PCR-table generation found
anywhere in the repository targeting the v3 production system specifically).
Evidence for this determination (not assumed from the filename):
- It is the only PCR implementation with an accompanying, passing test suite
  (`analysis/tests/test_pcr.py`,
  `test_final_paper_evaluation.py`).
- Its `config.py` imports condition/model identifiers directly from the frozen
  v3 registries (`whitebox_attack.v3.cues.V3_CONDITIONS`,
  `whitebox_attack.v3.production_models.FINAL_PRODUCTION_MODELS`) with
  assertions that fail loudly if those registries change shape — i.e. it is
  structurally pinned to the actual production configuration, not a
  hand-typed guess.
- Its output directory name (`v3_final_completed_qwen_llama_20260910`) and
  `README.md` explicitly document which prior directories it supersedes.
- The seven other `attack/*.py` scripts that also compute
  PCR-adjacent numbers (`generate_pcr_cdf_tables.py`,
  `evaluate_completed_shard.py`, `final_production_evaluation.py`,
  `qwen14b_architecture_mechanism_audit.py`) have no comparably recent,
  test-covered, registry-pinned output directory — see per-script notes below.

## Full-repository grep results

Searched terms: `PCR`, `pcr_at`, `premature`, `completion_by_step`,
`first_terminal`, `ResearchComplete`, `DIRECT_ANSWER`,
`researcher_no_tool_exit`, `post_branch`, `terminal_post_branch_index`,
`coverage_fraction`, `retrieval_complete` across the whole repository
(excluding `whitebox_attack/archive/`, per task constraint 4, and excluding
raw JSON/JSONL data files, which are data, not implementations).

## Per-implementation classification

| File | Classification | Notes |
|---|---|---|
| `analysis/pcr.py` | **Evidence-agnostic completion metric, TWO denominators, neither canonical alone** | See "Deep dive" below. This is the file this package reconciles against in `canonical_vs_previous_comparison.md`. |
| `analysis/final_paper_evaluation/termination_routes.py` | **Matches canonical intent closely, but folds MAX_BUDGET_EXIT into a broader `OTHER_VALID_TERMINATION`/`TRACE_LOGGING_FAILURE` split that does not isolate budget exhaustion** | Already keeps RESEARCH_COMPLETE/RESEARCHER_NO_TOOL_EXIT separate from TOOL_OR_PARSER_FAILURE and TRACE_LOGGING_FAILURE (its own docstring explicitly warns against combining "other" and "technical error" — a real, good practice this canonical package inherits). Does not have a dedicated MAX_BUDGET_EXIT label. |
| `analysis/final_paper_evaluation/mechanism.py` | **Uses `PCRRecord.terminal_post_branch_index` from `pcr.py` directly (not its own implementation)** | Downstream consumer, not a separate metric; inherits whichever definition `pcr.py` provides. |
| `analysis/final_paper_evaluation/recovery.py` | **Not a completion metric** — evidence/coverage-boundary and supervisor-redelegation classification only | Out of scope for PCR reconciliation; correctly kept separate from PCR per this task's Phase 7 requirement. |
| `attack/generate_pcr_cdf_tables.py` | **Uses the wrong denominator concept for "premature," combines technical failures with "other"** | Its own docstring states: *"This is a process-metric evaluator only. It never scores final-answer correctness, EM, or F1."* — that part is fine (matches the canonical independence requirement) — but its `terminal_category_counts` (lines ~1293-1305) explicitly folds any unrecognized channel **and** the count of `invalid_completed_rows` (execution failures) into a single `TERMINAL_OTHER_ERROR` bucket, i.e. it combines a genuine technical/execution failure with a merely-unusual-but-valid outcome. This is exactly the anti-pattern the canonical taxonomy is designed to avoid (task: "Do not fold maximum-budget exits, parser failures, trace failures, or execution failures into RESEARCH_COMPLETE or NO_TOOL_DIRECT_EXIT" — this script goes further and folds THEM INTO EACH OTHER). |
| `attack/evaluate_completed_shard.py` | **Evidence-gated completion metric — measures something different from the canonical PCR by design** | Its "premature completion" (`_premature_completion_step`) requires `coverage_fraction_after_observation < 1.0` at the `ResearchComplete` step — i.e. it is answering "did the researcher complete before evidence was sufficient," not "when did the researcher complete." This is a legitimate, separate, evidence-aware metric (kept under a different name, see Phase 7) — not a duplicate implementation of canonical PCR, and not directly comparable to it. |
| `attack/final_production_evaluation.py` | **Evidence-gated completion metric (different formula than `evaluate_completed_shard.py`)** | `premature_direct_answer`/`premature_research_complete` require `retrieval_required AND terminal_event_occurred AND NOT retrieval_complete_before_step` — a third, non-identical evidence-gated formula. Also collapses non-RC/non-DA cases into a single `"OTHER"` bucket with no separate technical-error category at all. |
| `attack/qwen14b_architecture_mechanism_audit.py` | **Closest existing analogue to the canonical route taxonomy, but qwen14b-only by construction** | Its own docstring/route enum (`ROUTE_RESEARCH_COMPLETE`, `ROUTE_RESEARCHER_NO_TOOL_EXIT`, `ROUTE_MAX_OR_BUDGET`, `ROUTE_OTHER_ERROR`) is the only pre-existing implementation that separates budget-exhaustion from technical error as this canonical package also does. It is explicitly scoped to `MODEL = "qwen14b"` only (never generalized to llama8b or a multi-model input), so it cannot itself serve as a cross-model canonical implementation. |
| `attack/rescore_completed_answer_v2.py` | **Not a completion metric** — group-level clean/attack accuracy comparison only | Out of scope; see Phase 7. |
| `attack/answer_evaluator_v2.py` | **Not a completion metric** — rule-based short-answer scoring | Out of scope; see Phase 7. Also the source of `official_em_f1`, a second, near-duplicate implementation of `attack/metrics.py::em_f1` (both SQuAD-style normalization, differing only in apostrophe-stripping order — functionally near-identical). |
| `attack/hop_aware_evaluation.py` | **Unknown / not a real implementation** — references `premature_completion` and `post_injection_search_count` as expected *pre-computed input field names*, computes none of them itself | This file is a generic per-field averaging utility; it does not implement PCR, it only assumes some upstream row already has a `premature_completion` boolean. |

## Deep dive: `analysis/pcr.py`

`PCRRecord.terminal_post_branch_index` (T_i) and `.terminal_route` are
computed via `analysis.trace_parser`/`trace_schema` — the
SAME parser this canonical package reuses (`canonical_completion.py` imports
`parse_trace_file` directly, does not reimplement trace walking). Therefore
**T_i and the RESEARCH_COMPLETE/NO_TOOL_EXIT determination are identical
between `pcr.py` and this canonical package for every row where both agree
an anchor exists** — confirmed empirically: `test_20_real_fixture...` in
`test_canonical_completion.py` directly cross-checks one real qwen14b/hotpot
production trace and passes.

The two implementations diverge only in **route taxonomy granularity** and
**eligibility (denominator) policy**:

1. **Route taxonomy**: `pcr.py` has exactly two positive routes
   (`ROUTE_RESEARCH_COMPLETE`, `ROUTE_NO_TOOL_EXIT`) and a residual
   `no_branch_reason` (`NO_SEARCH_ATTEMPTED` / `SEARCH_TOOL_ERROR` /
   `STRUCTURAL_PARSE_FAILURE`) plus an implicit "no terminal observed but
   branch reached" case folded into `route_summary_for_cell`'s `n_other`
   bucket. It has **no MAX_BUDGET_EXIT category** — an anchored episode that
   ran to the full 30-decision budget without ever resolving to RC/no-tool
   is indistinguishable, in `pcr.py`'s own output, from a genuine
   trace-logging gap. This canonical package splits that "no terminal
   observed" case into `MAX_BUDGET_EXIT` (decision count ≥ 30) vs.
   `TRACE_LOGGING_FAILURE` (decision count < 30, a real logging gap) —
   see `canonical_completion.py`'s `classify_local_terminal_route`.

2. **Denominator policy — a genuine internal inconsistency found in `pcr.py`
   itself**: its own module docstring (`pcr.py:19-22`) states rows with
   `branch_reached=False` "are excluded from BOTH denominators" (the
   completion-time CDF *and* the unconditional incidence). The
   **code** for `unconditional_incidence_for_cell` (`pcr.py:171-190`,
   `n = len(records)`) does **not** filter by `branch_reached` at all — it
   receives the FULL per-condition record list (including
   `NO_SEARCH_ATTEMPTED`/`SEARCH_TOOL_ERROR`/structural-failure rows) and
   uses its length as `n`. This contradicts the module's own top-of-file
   docstring, though it matches the SEPARATE Section-5.2-specific docstring
   a few lines later ("denominator = every valid completed-result row in
   the cell, regardless of outcome... rows with no observed terminal route,
   a technical parse failure, or no search ever attempted are visible mass,
   not hidden") — i.e. the file contradicts itself between its general
   framing paragraph and its Section-5.2-specific paragraph. This is
   reported here as a genuine, verifiable discrepancy (both quotes are
   exact, from the same file, `pcr.py` lines 19-22 vs. the Section 5.2
   docstring immediately following), not a criticism of intent — Section
   5.2's actual behavior (inclusive denominator) is a legitimate design
   choice, just inconsistently documented in the same file as a second,
   narrower ("both denominators") claim that isn't true of its own code.

3. **This canonical package's eligibility rule is a third, distinct, and
   explicitly documented policy** (see `canonical_completion.py`'s module
   docstring, "Eligibility (denominator)" section): `pcr_eligible =
   anchor_found AND route not in {TRACE_LOGGING_FAILURE, EXECUTION_FAILURE}`.
   This INCLUDES `MAX_BUDGET_EXIT` rows in the denominator (they had a valid
   anchor and are real "did not complete" information) but EXCLUDES rows
   with no anchor at all (no search ever attempted) and rows whose trace is
   simply unreadable. Numerically this sits between `pcr.py`'s two
   variants — see `canonical_vs_previous_comparison.md` for the exact,
   computed size of every difference.

## Files NOT read in full for this inventory (out of stated scope)

`analysis/final_paper_evaluation/{answer_harm.py,
figures.py, master.py, paper_tables.py}` — these consume `pcr.py`'s already-
computed `PCRRecord`s and evidence/recovery outputs downstream; they do not
implement a competing completion-step definition of their own (confirmed by
their imports, which pull `PCRRecord`/`EvidenceRecord` types rather than
recomputing terminal routes), so they are not separately classified above.

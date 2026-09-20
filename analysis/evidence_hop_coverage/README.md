# Evidence/hop coverage -- README

## Source artifacts (locked PCR)

This task's prompt names `MASTER_RESULTS.md` and `FULL_MULTISTEP_PCR_TABLES.md`
as the locked canonical PCR artifacts, and a `whitebox_attack/v3/` code
layout. Neither name/path exists verbatim in this repository at the frozen
commit -- both are pre-restructure names. The commit
`9c327466ab8e730ba29a71db375ea5aa7556981d` (this task's own frozen-lineage
commit) is *itself* the commit that renamed `whitebox_attack/v3/` to
`attack/`/`analysis/` (see its own message: "Restructure repo to
attack/analysis/results/archive"; the exact old->new path table is in
`analysis/canonical/reorg_prep/reorg_top_level.py`). Applying that
documented mapping (`whitebox_attack/v3/analysis/<x>` -> `analysis/<x>`)
places this task's required output directory at `analysis/evidence_hop_coverage/`,
used here.

For the PCR artifacts themselves: the only `MASTER_RESULTS.md` in this repo
is `qwen_llama_results_bundle/MASTER_RESULTS.md` (built earlier in this same
working session, from the identical qwen14b/llama8b production data this
task also targets). No file named `FULL_MULTISTEP_PCR_TABLES.md` exists
anywhere in the repo; the closest and evidently-intended match is
`qwen_llama_results_bundle/tables/full_pcr_tables.md` ("Full PCR tables,
every step 1-29, plus PCR@30"). This README treats those two files as the
locked prior work referred to by the task prompt. **This is a reported
interpretation, not a silent guess** -- if a different `MASTER_RESULTS.md`/
`FULL_MULTISTEP_PCR_TABLES.md` was intended, this run should be redone
pointing at it.

Both of those files -- and every completion field in this package's own
output -- are generated from `analysis/canonical/canonical_completion.py`,
which is unmodified by this task.

## Commit lineage

- Frozen commit required by the task: `9c327466ab8e730ba29a71db375ea5aa7556981d`
- Current `HEAD`: `9c327466ab8e730ba29a71db375ea5aa7556981d`
- `HEAD == frozen commit`: **True**
- Working tree has uncommitted changes on top of that commit:
  **True** (expected -- this
  session's own `qwen_llama_results_bundle/` and `analysis/evidence_hop_coverage/`
  outputs are new, uncommitted files; no tracked file was modified or reset).

No `git reset`/`checkout` was performed. If `HEAD` above does not equal the
frozen commit when you read this, the production data may have changed
since this run and the run should be repeated.

## Regenerate

    conda activate whitebox
    python -m analysis.evidence_hop_coverage.build_evidence_hop_coverage \
        --output-dir analysis/evidence_hop_coverage \
        --pcr-bundle-dir qwen_llama_results_bundle/data

## Dataset evidence mappings (spec section "Canonical evidence mapping")

Required evidence per example is read from each trace's own
`required_paragraph_ids` / `hop_count` / `required_dependency_count` fields
-- these are written once, at production-trace-generation time, by
`attack/runtime/corpus.py` (`build_hotpot_rows` / `build_2wiki_rows` /
`build_musique_rows` / `build_multihoprag_rows`), directly from each
dataset's own annotations (HotpotQA `supporting_facts`, 2WikiMultiHopQA
`supporting_facts`, MuSiQue `question_decomposition` +
`paragraph_support_idx`, MultiHopRAG `evidence_list`) -- never inferred from
a generated answer, and frozen into the trace at that time (this task does
not reload the source HF datasets or rerun any model).

| Dataset | Stratum field | Values seen | Hop order source |
|---|---|---|---|
| HotpotQA | `hop_count` (fixed) | {2} | supporting_facts order (task: treat as two-hop, not stratified) |
| 2WikiMultiHopQA | `hop_count` | {2, 4} | supporting_facts order |
| MuSiQue | `hop_count` | {2, 3, 4} | `question_decomposition` order (official) |
| MultiHopRAG | `required_evidence_count` (task: not an ordered hop chain) | {2, 3, 4} | annotation list position only -- explicitly not validated as reasoning order |

Hop index within an example = position in the trace's own
`required_paragraph_ids` tuple. Verified against real traces (not assumed)
that this tuple is stable and identically ordered across every condition of
the same `(model, dataset, example_id)` (spot-checked 3 conditions of one
hotpot example, byte-identical list).

## Evidence matching rules

- Matching is exact-ID-set intersection against
  `required_paragraph_ids`/`required_dependency_ids` as stored in the
  trace -- never fuzzy text matching, never inferred from the generated
  answer.
- Coverage at the **anchor** (targeted observation preceding post-observation
  step 1) reuses the trace's own `coverage_fraction_at_injection` /
  `dependencies_seen_at_injection` / `missing_dependency_ids_at_injection`
  fields verbatim (computed once, at the single anchor search event, by the
  production runner -- `attack/runtime/evidence.py::evaluate_coverage`).
- Coverage at **local completion (T_i)** reuses
  `analysis.evidence._cumulative_union_coverage`'s method (a raw union of
  every search event's `retrieved_paragraph_ids` within episode 0), which
  that module's own docstring documents as necessary rather than trusting a
  single "seen so far" snapshot field (verified there to lose a previously-
  found dependency in some llama8b multi-search episodes). Episode 0 ends,
  by construction, at its own terminal decision, so this union already
  excludes anything retrieved after T_i -- there is no separate "step
  budget" to slice.
- Coverage at **final workflow termination** is a NEW computation added by
  this script: the same raw-union method, applied across every episode in
  the trace (episode 0 plus any supervisor redelegation), not just episode
  0. This is deliberately not the trace's own `coverage_fraction_at_termination`
  telemetry field (kept in `analysis.evidence` as
  `whole_run_cumulative_retrieval_coverage_fraction` but not reused here) --
  computing final coverage with the exact same method as local coverage
  guarantees a "final < local" reading can only reflect a real trace
  anomaly, never a difference in measurement method. This is also why gate
  9 (final >= local) is structurally almost tautological except where a
  trace anomaly exists -- which is exactly what that gate is for.

## Reconciliation with `qwen_llama_results_bundle`'s evidence tables

`qwen_llama_results_bundle/data/evidence_summary.csv` (and its
`tables/evidence_tables.md`) also reports a "paired evidence loss" number
for the same comparisons/models/datasets. **Those numbers will not match
this package's `paired_evidence_deltas.csv`, by design, not by bug** --
they measure a different boundary:

| | `qwen_llama_results_bundle/data/evidence_summary.csv` | This package's `paired_evidence_deltas.csv` |
|---|---|---|
| Boundary | Final-report-input coverage (`EvidenceRecord.final_input_coverage_fraction`) | Local completion at T_i (`local_coverage_fraction`) |
| Method | Citation-ID extraction from `final_report_generation`'s actual input text | Raw retrieved-paragraph-ID set intersection |
| When | Downstream -- after any supervisor redelegation and compression | Upstream -- at the researcher's own first post-observation completion decision |

**Per-row cross-check performed**: `qwen_llama_results_bundle/data/evidence_run_level.csv::local_completion_coverage`
is actually built from `EvidenceRecord.episode0_terminal_coverage_fraction`
-- the SAME union-of-episode-0-search-events method this package's
`local_coverage_fraction` uses, reused from the same
`analysis.evidence._cumulative_union_coverage` function. Comparing the two
columns row-by-row across all 42,547 valid rows: **42,155 (99.1%) match
exactly**; the remaining 392 are rows where `evidence_run_level.csv` reports
a real number but this package reports `NA` -- in every one of those cases
the row's terminal route is NOT a genuine PCR completion route (mostly
`MAX_BUDGET_EXIT`). `evidence_run_level.csv`'s field is defined for any
episode with a search at all, regardless of whether it ever reached a
completion decision; this package deliberately restricts `local_coverage_fraction`
to `NA` unless `T_i` is actually defined (spec: "For MAX_BUDGET_EXIT runs:
... set completion-specific fields to missing/NA rather than zero"). This
confirms the two packages' underlying retrieval accounting is consistent --
the packages differ only in (a) which downstream/upstream boundary their
PAIRED summary reports, and (b) how strictly they gate that boundary to
`NA` for non-completing runs.

**When citing evidence numbers, name the boundary explicitly** ("evidence
retrieved by local completion" vs "evidence present in the final report
generator's input") rather than saying "evidence coverage" alone -- see
`qwen_llama_results_bundle/tables/TABLE_INDEX.md`'s matching
disambiguation section.

## Denominators

See `EVIDENCE_HOP_RESULTS.md` section 2 and `evidence_hop_summary.csv`'s
`n_valid` / `n_evidence_eligible` / `n_completed_and_evidence_eligible` /
`n_max_budget_exit` columns. `evidence_eligible` = valid run AND
`required_evidence_count > 0`. `completed_and_evidence_eligible` additionally
requires `pcr_eligible` AND a defined `T_i` (terminal route is
RESEARCH_COMPLETE or NO_TOOL_DIRECT_EXIT).

## Reused implementations

- `analysis.canonical.canonical_completion` -- PCR (T_i, terminal_route,
  pcr_eligible, anchor fields). Unmodified.
- `analysis.completion_inventory` -- run inventory, valid-row rule
  (`_row_is_valid`). Unmodified.
- `analysis.trace_parser` / `analysis.trace_schema` -- all trace/event
  parsing (episodes, decisions, search events). Unmodified.
- `analysis.evidence._cumulative_union_coverage` / `._paragraph_index` --
  reused directly (imported, not copied) for local-completion coverage and
  ID normalization.
- `analysis.canonical.results_bundle.comparisons.COMPARISONS` -- the single
  reconciled condition-comparison manifest (built earlier this session to
  replace two previously-inconsistent lists). No second comparisons list is
  introduced here.
- `analysis.canonical.canonical_completion.build_strict_pairs` -- paired-
  eligible pairing rule (excludes rather than imputes).
- `attack.runtime.corpus.py` / `attack.runtime.evidence.py` -- read (not
  imported at runtime) to confirm where `required_paragraph_ids`/`hop_count`
  originate and that they are derived from dataset annotations, never from
  a generated answer.

Nothing in `analysis/evidence.py`, `analysis/canonical/canonical_completion.py`,
`analysis/trace_parser.py`, or `analysis/canonical/results_bundle/` was
modified by this task.

## Known limitations

1. **Sentence-level supporting-fact recall (HotpotQA/2WikiMultiHopQA) is not
   computed.** `attack/runtime/corpus.py` captures `sentence_ids` per
   dependency at corpus-build time, but this is not one of the fields
   serialized into the frozen trace JSON (only paragraph/document-level
   `required_paragraph_ids` and dependency-level
   `dependencies_seen_at_injection`/`missing_dependency_ids_at_injection`
   are). Recovering sentence-level recall would require reloading the
   original HuggingFace dataset objects and re-running `build_hotpot_rows`/
   `build_2wiki_rows` outside the trace -- out of scope for this pass per
   the task's own tight scope constraints. Document-level (paragraph)
   coverage, which the task also asks for and treats as the primary unit,
   is fully computed.
2. **MuSiQue reasoning-path text (`decomposition_question`/
   `intermediate_answer`/`bridge_entities`) is not recovered**, for the same
   reason (computed by `corpus.py` at build time, not serialized into the
   trace). Hop **order** and hop **count** ARE recovered reliably (position
   in `required_paragraph_ids`, verified stable across conditions), just not
   the natural-language decomposition text itself.
3. **Compressed-notes and final-report-input evidence boundaries** (spec
   3.2's "final-run coverage" via what `final_report_generation` actually
   received, as opposed to what was merely retrieved) are available in
   `analysis.evidence.EvidenceRecord` but are NOT joined into this task's
   run-level table, since this task's "final workflow termination" is
   explicitly about retrieval state, not generator-input visibility (a
   separate, later analysis dimension per the task's own stop-list: "Do not
   ... calculate final-answer harm").
4. **Oracle Complete anchor-coverage gate (13)**: reported, not forced --
   see `EVIDENCE_HOP_RESULTS.md` section 7 for the actual pass/fail result
   and, if it fails, the specific rows involved.
5. This package's own `n_records` at generation time was **43200**
   canonical run rows (2 models x 4 datasets x 12 conditions); see
   `run_level_evidence_hops.csv` for the frozen count at whatever time you
   are reading this.

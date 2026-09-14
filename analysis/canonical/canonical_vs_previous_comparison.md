# Canonical vs. previous implementation: reconciliation (Phase 6)

Comparison is against `analysis/pcr.py`'s already-generated
outputs at
`whitebox_paper_results/e2e_full_system_v3/production/evaluation/v3_final_completed_qwen_llama_20260910/pcr/`
(2026-09-10), determined in `legacy_metric_inventory.md` to be the actual
latest v3 result-producing implementation (not the HEAD-commit "PCR 40-step"
legacy pipeline, which is a different, pre-v3 system explicitly excluded per
task constraint 4).

**Granularity caveat, stated up front:** `pcr.py`'s CSVs are pre-aggregated
per (model, dataset, condition) cell — no `example_id` is retained in its
output. A true row-level diff (per the task's requested
`canonical_vs_previous_row_differences.csv` filename) is therefore a
**cell-level** diff here, computed by `reconcile.py`, not a per-example diff.
This is stated explicitly rather than silently implied.

## Headline answer: did the latest results already match the canonical metric?

**Partially, and only for one of the two denominators `pcr.py` computes.**
`pcr.py` produces two different tables per cell (Section 5.1
"completion-time CDF" and Section 5.2 "unconditional incidence"), and never
designates one as canonical. Compared against this package's single
canonical PCR@k (96 cells, `reconcile.py` output):

| Comparison | Result |
|---|---|
| Cells where canonical eligible-N differs from `pcr.py`'s Section 5.1 denominator | 19 / 96 |
| Max absolute PCR@k difference vs. Section 5.1 (`completion_cdf`), any of k=1,3,5 | 0.52 percentage points |
| Cells where PCR@1 differs from Section 5.1 by more than 0.05pp | **0 / 96** |
| Cells where PCR@1 differs from Section 5.2 (`unconditional_incidence`) by more than 0.05pp | 63 / 96 |
| Max absolute PCR@k difference vs. Section 5.2, any of k=1,3,5 | 33.1 percentage points |

**Interpretation:** the canonical PCR@k is numerically very close to `pcr.py`'s
Section 5.1 ("completion-time") variant at early checkpoints — this is
expected and correct, because Section 5.1's denominator (branch-reached rows
with an observed terminal event) and this package's eligible set overlap for
the vast majority of rows; they diverge only on the ~19/96 cells containing
at least one `MAX_BUDGET_EXIT` row, which Section 5.1 silently excludes from
its denominator (making its curve artificially reach 100% once all
non-budget-exhausted rows have completed) while the canonical definition
keeps such rows in the denominator forever unresolved (so the canonical
curve can plateau below 100%, a deliberate and disclosed design choice — see
`canonical_completion.py`'s module docstring). Section 5.2's much larger
divergence (up to 33 percentage points) is because its denominator includes
`NO_SEARCH_ATTEMPTED`/`SEARCH_TOOL_ERROR`/structural-failure rows that have
no defined anchor at all — a fundamentally different, non-comparable
population, not a computational bug in either implementation, but a policy
choice this canonical package deliberately does not adopt (see rationale in
`canonical_completion.py`).

## Rows classified identically

For every cell, the canonical `n_RESEARCH_COMPLETE` + `n_NO_TOOL_DIRECT_EXIT`
counts that ARE anchor-eligible match `pcr.py`'s Section 5.1 completion-time
denominator exactly (spot-checked: llama8b/2wiki/NaturalClean — canonical
anchored-RC+NT = 429, `pcr.py`'s `n_completion_cdf_denominator` = 429,
exact match). This is expected: both implementations call the identical
`parse_trace_file`/`ResearcherEpisode.terminal_post_branch_index` code path
(this package imports it directly rather than reimplementing it), so wherever
both implementations agree an anchor exists and a terminal route was
observed, they agree on T_i and the route by construction.

## Rows with a different observation anchor

None found. Both implementations use the identical anchor definition (first
executed `benchmark_search` event in episode 0) because both call the same
`ResearcherEpisode._branch_decision_index`/`.branch_reached` property. No
divergence is possible here since neither implementation independently
re-derives the anchor.

## Rows with a different step index

None found for any row where both implementations classify a route (see
`test_canonical_completion.py::test_20`). T_i is computed by the same shared
`terminal_post_branch_index` property in both cases.

## Rows with a different terminal route

**Three per cell, on average, in the specific sense of "no-search-but-
declared-complete" rows** (spot-checked: llama8b/2wiki/NaturalClean has
exactly 3 such rows). `pcr.py` labels every `branch_reached=False` row's
`terminal_route` as `None` unconditionally (since its `PCRRecord.terminal_route`
reuses the POST-BRANCH-scoped `ResearcherEpisode.terminal_route`, which is
always `None` when there is no branch), and buckets them all under
`no_branch_reason=NO_SEARCH_ATTEMPTED`, contributing to `n_other_terminal_route`
in its route summary. This canonical package instead reports what actually
happened using the WHOLE-EPISODE-scoped `overall_terminal_route` for these
specific rows (`classify_local_terminal_route`'s first branch) — i.e. a row
where the researcher called `ResearchComplete` without ever searching is
labeled `RESEARCH_COMPLETE` here (richer information), **but is still
excluded from `pcr_eligible`** exactly as `pcr.py` excludes it from both of
its denominators. This is a genuine labeling difference, not a numerator
difference — no such row is ever counted in either implementation's PCR
percentage.

## Rows previously counted as completion but now classified as budget or technical failure

None. No row that `pcr.py` counts as `ROUTE_RESEARCH_COMPLETE`/`ROUTE_NO_TOOL_EXIT`
is reclassified as `MAX_BUDGET_EXIT`/`TOOL_OR_PARSER_FAILURE` by this
package — the reclassification only ever moves rows FROM `pcr.py`'s
undifferentiated "other terminal route" / "no terminal observed" bucket INTO
the new `MAX_BUDGET_EXIT` category (never out of a completion category into
a failure category).

## Rows newly recognized as no-tool/direct exits

None beyond the "no-search-but-declared-complete" relabeling described above
(some of which resolve to `NO_TOOL_DIRECT_EXIT` rather than
`RESEARCH_COMPLETE`, per whichever the episode's own `overall_terminal_route`
actually was).

## Changes in PCR@k

See the headline table above and `canonical_vs_previous_row_differences.csv`
for the full per-cell breakdown at k=1,3,5 against both legacy variants.

## Changes in eligible N

19/96 cells differ from `pcr.py`'s Section 5.1 denominator (always by the
count of `MAX_BUDGET_EXIT` rows added back in, minus any
no-search-but-declared-complete rows also present in that cell — both counts
are directly visible in `canonical_route_decomposition.csv`). All 96 cells
differ from Section 5.2's denominator, usually substantially, because
Section 5.2 never excludes `NO_SEARCH_ATTEMPTED`/`SEARCH_TOOL_ERROR` rows at
all.

## Changes in paired N

Not recomputed identically here — `pcr.py` does not itself compute paired
deltas (that is `analysis/pairing.py`'s job, reused
unmodified as the pairing invariant by both the pre-existing analysis
package's `strict_pairing_audit.csv` and this package's
`retrieval_equivalence.py`). The existing `strict_pairing_audit.csv`
(77,411 condition-pair-example rows, all valid qwen14b/llama8b examples)
shows only **34.8%** of all condition-pairs pass the strict
`validate_paired_traces` invariant (identical `shared_prefix_hash` and
`base_tool_message_hash`) — i.e. most pairs in production do NOT share an
identical pre-injection retrieval prefix. This package's own
`canonical_paired_condition_deltas.csv` computes PCR@k deltas over exactly
the `pcr_eligible`-intersected example set per pair (not the strictly-hash-
matched set), which is a different, and generally larger, notion of "paired"
than the hash-strict one — this distinction is stated explicitly rather than
conflated: `build_strict_pairs` in `canonical_completion.py` pairs on
`model+dataset+example_id` and both-sides-eligible, which is necessary for a
PCR delta to be computable at all, but does NOT itself verify retrieval-prefix
identity — that is exactly what `retrieval_equivalence_summary.md` (Phase 9D)
separately measures.

## Conclusion

The latest pre-existing v3 result-producing implementation (`pcr.py`) does
**not** already match the canonical metric exactly, but is very close to it
under one of its two internally-inconsistent denominator choices (Section 5.1)
and substantially different under the other (Section 5.2). No row is
reclassified from a completion route into a failure route or vice versa by
this canonicalization — every difference is either (a) a denominator-policy
difference affecting the PCR@k percentage at higher k values in cells with
budget-exhausted rows, or (b) a labeling-richness difference for rows that
were already excluded from both implementations' PCR calculations. These are
described as corrections to ambiguity and inconsistency, not as
"improvements" to the underlying finding — the tests in
`test_canonical_completion.py` are what establish correctness, not this
comparison alone.

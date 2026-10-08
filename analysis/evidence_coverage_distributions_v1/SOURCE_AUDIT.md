# Source Audit — Evidence Coverage Distribution Analysis

Date: 2026-10-06. Read-only audit against canonical sources; nothing in
`analysis/evidence_hop_coverage/`, any raw trace, `MASTER_RESULTS.md`, or any
PCR output was modified to produce this audit.

## 1. Sources located

| Model | File | Status |
|---|---|---|
| Qwen2.5-14B-Instruct | `analysis/evidence_hop_coverage/run_level_evidence_hops.csv` | **Usable directly** — correctly gated, internally consistent |
| Llama-3.1-8B-Instruct | same file | **Usable directly** |
| GPT-OSS-20B | `gemma_gptoss_results_bundle/data/evidence_run_level.csv` | **Usable only after a required join + filter + field substitution** — see §3 below |

`gemma_gptoss_results_bundle/` also contains Gemma-3-27B-IT rows in every
file above; every row read from it in this package is filtered to
`model == "gpt_oss_20b"` before any other processing. No Gemma value
appears anywhere in this package's outputs.

## 2. Qwen/Llama: checks 1–10

1. **Uniqueness** — 43,200 rows, 43,200 unique `(model, dataset, example_id, condition)` keys. PASS.
2. **Coverage in [0,1]** — 0 out-of-range values across 43,200 rows. PASS.
3. **Retrieved ≤ required** — enforced structurally (`local_evidence_count_retrieved = len(local_retrieved_ids)`, a subset of the required set); 0 violations found by direct check. PASS.
4. **Coverage = retrieved/required** — checked directly: `local_evidence_count_retrieved / required_evidence_count == local_coverage_fraction` for all 42,143 evidence-eligible, non-null rows, 0 violations. PASS.
5. **Full coverage ⇔ coverage = 1** — `local_full_evidence = (local_coverage_fraction >= 1.0)`, by construction in `build_evidence_hop_coverage.py:284`. PASS.
6. **Missing = required − retrieved** — checked directly alongside #4, 0 violations. PASS.
7. **No post-completion evidence in local coverage** — confirmed by reading `build_evidence_hop_coverage.py:265-284`: `local_coverage_fraction` is computed **only** from episode 0's own search-event union, and **only when `completed_within_budget` is True** (`terminal_route in {RESEARCH_COMPLETE, NO_TOOL_DIRECT_EXIT}` — `canonical_completion.PCR_NUMERATOR_ROUTES`). For any other terminal route (including `MAX_BUDGET_EXIT`), `local_coverage_fraction` is explicitly `None`, never a fabricated 0 or a value computed from later search activity. This is the single most important correctness property this file has, and it holds by construction, not by accident.
8. **Visible denominators** — every output table in this package carries `eligible_N` beside every statistic; see `coverage_distribution_by_condition.csv`.
9. **Paired comparisons use identical example IDs** — enforced in `pairing.py::pair_by_example_id`, which intersects `(dataset, example_id)` sets between the two conditions being compared; never pads or imputes a missing partner.
10. **Dataset-separated, not reconstructed from averages** — confirmed: the file carries a `dataset` column with real per-row values for all 4 datasets; no averaging was ever applied upstream of this file.

**No blocker for Qwen/Llama.**

## 3. GPT-OSS: a real field-semantics trap, found and fixed before use

This is the one genuine problem this audit found, and it would have silently
produced wrong numbers if not caught here.

### 3a. `missing_hop_count` / `full_hop_indicator` are the WRONG boundary

Reading `analysis/canonical/results_bundle/evidence_ext.py:38-70` (the
module that builds `evidence_run_level.csv` for *both* the Qwen/Llama and
GPT-OSS bundles) shows:

```python
"local_completion_coverage": r.episode0_terminal_coverage_fraction,   # correct boundary (T_i)
"final_run_coverage": r.final_input_coverage_fraction,                 # final-report-input boundary
"missing_hop_count": r.final_input_missing_count,                      # <- FINAL boundary, not local!
"full_hop_indicator": r.full_hop_coverage_at_final_input,              # <- FINAL boundary, not local!
```

`missing_hop_count` and `full_hop_indicator` are computed at the
**final-report-generation-input** boundary (`RESULTS_CANONICAL_INDEX.md`
§5's "downstream" boundary), **not** the local-completion (T_i) boundary
that `local_completion_coverage` itself uses. Verified numerically: of
15,161 `gpt_oss_20b` rows, 1,704 have `local_completion_coverage !=
final_run_coverage` (i.e. the two boundaries genuinely disagree for that
row); for the first 5 inspected, `missing_hop_count` matched
`required − (final_run_coverage × required)`, not `required −
(local_completion_coverage × required)` — e.g. one row has
`required=2, local_completion_coverage=0.5, final_run_coverage=1.0,
missing_hop_count=0` — `missing_hop_count=0` is consistent only with the
**final** boundary (`2 − round(1.0×2) = 0`), not the **local** boundary
(`2 − round(0.5×2) = 1`).

**Consequence**: `missing_hop_count` and `full_hop_indicator` from
`evidence_run_level.csv` are **never used anywhere in this package**. This
package instead derives a local-boundary missing count and full-coverage
flag directly from `required_dependency_count` and `local_completion_coverage`
(both of which *are* at the correct, local-completion boundary) —
`retrieved = round(local_completion_coverage × required_dependency_count)`,
`missing = required_dependency_count − retrieved`, `full = (missing == 0)`.
This is a derivation from already-computed canonical run-level fields, not
a new recomputation from traces.

### 3b. `local_completion_coverage` itself is not pre-filtered to genuine PCR completions

Reading `analysis/evidence.py:248-258` (`build_evidence_record`, the shared
function both bundles' `evidence_run_level.csv` files are built from) shows
`episode0_terminal_coverage_fraction` is computed whenever episode 0 had
**any** search activity at all (`ep0_had_search`), with **no check against
`terminal_route`**. This is the opposite of the Qwen/Llama-side package's
own `completed_within_budget` gate (§2 check 7 above) — GPT-OSS's
`evidence_run_level.csv` includes coverage values for rows that never
reached a genuine PCR-qualifying completion (`MAX_BUDGET_EXIT`,
`TOOL_OR_PARSER_FAILURE`, `EXECUTION_FAILURE`, etc.), which is exactly the
population this task's Canonical Evidence Variable excludes by definition.

Quantified: `gemma_gptoss_results_bundle/data/pcr_run_level.csv` (gpt_oss_20b
rows only) has `terminal_route` ∈ {NO_TOOL_DIRECT_EXIT: 12,124,
RESEARCH_COMPLETE: 2,996, EXECUTION_FAILURE: 965, TRACE_LOGGING_FAILURE: 24,
MAX_BUDGET_EXIT: 17}. Of the 15,161 rows in `evidence_run_level.csv`, 41
(0.27%) correspond to an `(dataset, condition, example_id)` key whose
`terminal_route` is **not** `RESEARCH_COMPLETE` or `NO_TOOL_DIRECT_EXIT`.

**Fix applied**: every GPT-OSS row used anywhere in this package is joined
against `pcr_run_level.csv` on `(dataset, condition, example_id)` and kept
only if `terminal_route ∈ {RESEARCH_COMPLETE, NO_TOOL_DIRECT_EXIT}`. The 41
non-qualifying rows are excluded, not included with a caveat — they are not
a measurement of "coverage at completion" at all, since no completion
occurred. This join is logged in `coverage_distribution_by_condition.csv`'s
row counts and in `validation_gates.csv`.

### 3c. Retrieved-count derivation validity check

Since `evidence_run_level.csv` carries no explicit retrieved-count column
for GPT-OSS, `retrieved = round(local_completion_coverage ×
required_dependency_count)` must itself be validated: coverage should
always be (very close to) an exact ratio of two small integers. Checked:
after the join-filter in §3b, every row's
`abs(local_completion_coverage × required_dependency_count − round(...))
< 1e-6` — see `validation_gates.csv` gate 4. 0 violations found.

## 4. Dataset coverage

| Model | HotpotQA | 2WikiMultiHopQA | MuSiQue | MultiHopRAG |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | complete | complete | complete | complete |
| Llama-3.1-8B-Instruct | complete | complete | complete | complete |
| GPT-OSS-20B | complete | **incomplete — 38/108 shards as of 2026-10-06; a rerun is in progress** | complete | complete |

Per instruction, GPT-OSS/2WikiMultiHopQA is **not** derived from a
macro-average or any other dataset — every GPT-OSS figure's 2WikiMultiHopQA
panel displays "Dataset-level run data unavailable" and no number for that
cell appears in any CSV.

## 5. No blocker

A genuine field-semantics problem was found for GPT-OSS (§3), but it has a
disclosed, derivable-from-canonical-data fix (join + substitute two
columns), not a missing-data blocker. GPT-OSS/2WikiMultiHopQA is a real,
separate, already-known gap (§4), handled by omission, not estimation.
This package proceeds with all three models, with GPT-OSS's known 3-of-4
dataset scope and the §3 field substitution applied throughout.

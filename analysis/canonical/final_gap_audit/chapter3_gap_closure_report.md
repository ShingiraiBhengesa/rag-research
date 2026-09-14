# Chapter 3 gap closure report (Phase 10)

Final status: **READY WITH DISCLOSED LIMITATIONS**

This reflects the task's own explicit guidance: the previously-unknown
upstream ODR commit is not a blocker for readiness (it is now resolved
anyway, see below) — the real remaining limitation is the missing thesis
project itself, which blocks only the mechanical act of writing text into
Chapter 2/3 files that do not exist here, not the completeness of the
underlying evidence base.

## 1. Is the system replication/adaptation description complete?

**Yes.** `chapter3_final_source_of_truth.md` §1-5 plus
`chapter3_source_of_truth.md` §A give a full node-by-node, file-by-file
comparison against upstream ODR, and the previously-`UNKNOWN` exact upstream
commit is now resolved: `1f24f1142db24f81e29eb88751985a00ec8ed580`
(`upstream_odr_recovery.md`, `EXACT COMMIT VERIFIED`, very high confidence
from 7 byte-identical files and ~15 converging dependency-version pins).

## 2. Is the attack implementation fully documented?

**Yes.** Exact insertion point, injection gate, one-shot counter, and
verbatim cue strings for all 12 conditions are documented in both prior
audits and unchanged by this one.

## Oracle availability (Phase 6 addendum)

Both `OracleComplete_*` conditions are confirmed present for **all four**
datasets including MultiHopRAG (`final_condition_matrix.csv`), and a direct
scan of every oracle-mode search event in the completed qwen14b/llama8b
production data (`oracle_availability_summary.csv`, 16 model×dataset×
oracle-condition combinations, ~7,100 oracle search events total) found
**zero** `oracle_complete_available=False` cases — the oracle-unavailability
fallback path (`_oracle_unavailable_message`) exists in code but was never
actually triggered in the completed production data for either model.

## 3. Are all twelve conditions documented?

**Yes**, verbatim, in `condition_metadata.csv` and
`chapter3_final_source_of_truth.md` §8-9.

## 4. Are the nine primary and three secondary labels consistent?

**Yes**, internally consistent across `condition_metadata.csv`,
`chapter3_final_source_of_truth.md`, and the route-breakdown outputs in this
final gap audit (`route_breakdown_primary_conditions_only.csv`/
`route_breakdown_secondary_ablations_only.csv` both computed from the same
grouping). Internal condition identifiers were never renamed.

## 5. Is PCR fully defined and tested?

**Yes.** Definition unchanged from `chapter3_final_source_of_truth.md`
(this audit was explicitly instructed not to change it, and did not). 20/20
unit/regression tests pass (`test_canonical_completion.py`). This audit adds
an empirical cross-check specific to the step-budget boundary: the maximum
observed post-observation step (29) exactly equals the theoretical maximum
implied by the 30-decision budget, independently re-confirmed here via the
route-breakdown checkpoint identities (`PCR@k == RC_rate@k + NT_rate@k`,
verified with **zero failures across all 768 (model,dataset,condition,k)
combinations**, tolerance 1e-6).

## 6. Are `ResearchComplete` and no-tool/direct exits separately measurable?

**Yes**, and now measured at every requested granularity
(`route_breakdown_overall/by_model/by_dataset/by_condition/by_checkpoint/
by_model_dataset_condition/primary_conditions_only/secondary_ablations_only.csv`).
Overall, among 41,104 PCR-eligible runs: at k=30, ResearchComplete accounts
for 75.6% (31,077 runs) and NoToolDirectExit for 24.3% (9,999 runs) of
completions — roughly a 3:1 ratio.

## 7. Is the statement that no-tool exits are usually direct answers supported?

**No, and this is now backed by a full content audit, not just a route
count.** See `direct_exit_audit_report.md` in full. Headline: even under the
most generous PRELIMINARY (unconfirmed, pending human adjudication)
classification, only 31.2% of the 10,003 `NO_TOOL_DIRECT_EXIT` responses
resemble a concise direct answer, while 64.4% more closely resemble an
extended research summary — in every model, dataset, and step-bucket
breakdown. For Llama-3.1-8B specifically, 47.6% of its immediate step-1
no-tool exits are demonstrably malformed/failed tool-call attempts (a
DETERMINISTIC finding, not preliminary), not answers of any kind. **The
"usually direct answers" framing is not defensible** and Chapter 3 must not
use it; see `direct_exit_audit_report.md`'s exact supported wording (its
answer to question 7).

## 8. Are dataset sampling and selection fully documented?

**Yes, and now with exact eligible-population counts** (previously
`UNKNOWN`, now resolved in `eligible_population_counts.csv`/
`sampling_reproduction_check.md`): HotpotQA 7,405/7,405 eligible (0
rejected), 2WikiMultiHopQA 12,576/12,576 (0 rejected), MuSiQue 2,417/2,417
(0 rejected), MultiHopRAG 2,255/2,556 eligible (301 rejected, all via the
`multihoprag_oracle_not_allowed`/no-corpus-documents filter chain).
**Exact sampling reproduction PASSED for all 4 datasets** — reproducing the
recorded seed (20260820) and sampling method against the same cached
dataset files yields an identical example-ID set, identical order, and
identical `eligible_population_sha256`/`ordered_sample_id_sha256` digests
to the ones already embedded in production provenance.

## 9. Is the model set final?

**No, and this must be stated plainly, not implied.** Only `qwen14b` and
`llama8b` are `INCLUDED_COMPLETE`. `gemma27b` is `PARTIAL_ACTIVE` (414/432
shards, job genuinely still running). `gpt_oss_20b` is `BLOCKED` (0/432,
permanently stuck on a dead SLURM dependency pending a human decision).
`qwen7b` is `VALIDATION_ONLY` by original design. See
`final_model_status.md`/`.csv`.

## 10. Is the thesis LaTeX project ready for Chapter 3?

**Cannot be determined — no thesis project exists on this machine.** See
`thesis_project_inventory.md`. This is a genuine blocker for the literal
"write into the LaTeX files" step, but not for content-readiness: every
piece of evidence Chapter 3 needs is fully assembled and cited in
`chapter3_final_source_of_truth.md` plus this final gap audit.

## 11. Are all required citations present and verified?

**No bibliography exists to check "presence" against**, but every citation
Chapter 3 will need has been independently verified against a primary
source (arXiv, ACL Anthology, MIT Press TACL, or the relevant official
GitHub repository) and is ready to paste into a `references.bib` once one
exists: see `chapter3_citation_audit.md` and `chapter3_missing_citations.bib`
(10 verified entries: Open Deep Research, LangGraph [software, no paper],
HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHopRAG, Qwen2.5, Llama 3.1,
Gemma 3, BM25).

## 12. What exact items, if any, still block writing?

1. **No thesis LaTeX project exists on this machine** — the single true
   blocker for the mechanical act of writing. Resolve by locating/providing
   the actual project (a different machine, an Overleaf link, or a path on
   this machine this audit did not search).
2. **Chapter 2's actual wording could not be checked** — `chapter2_required_corrections.md`
   provides the exact search patterns and a ready-to-adapt replacement
   sentence template, but the real sentence(s), if any, have not been
   located or quoted.
3. **The direct-exit content classification is preliminary for 95.8% of
   rows** — the deterministic findings (malformed-tool-attempt rates,
   refusal/empty/error rates) are solid and can be cited now; the
   direct-answer-vs-research-summary split should be described as
   preliminary pending the human adjudication of
   `direct_exit_human_review_sample.csv`, OR Chapter 3 can simply cite the
   weaker, fully-supported claim (research summary is more common than
   direct answer, in every breakdown) without needing exact percentages
   confirmed.
4. **`gemma27b`/`gpt_oss_20b` are not part of the final analysis** — this is
   not a blocker to writing (the thesis can and should state the final
   model set is `qwen14b`+`llama8b`, with the other two explicitly disclosed
   as partial/blocked), but it must be stated as a disclosed limitation, not
   omitted.

**None of these four items requires further code changes, reruns, or
experiment modification to resolve** — items 1-2 require access to a
different file/machine; item 3 requires human time, not more computation;
item 4 is a factual disclosure, not a gap in the underlying evidence.

## Explicit non-blocking clarification (per task instruction)

The previously-unknown upstream ODR commit is **not** treated as a
readiness blocker — and it is now resolved anyway
(`upstream_odr_recovery.md`, `EXACT COMMIT VERIFIED`). Both before and after
this resolution, the local package version (`0.0.16`) and the fully
enumerated local modifications were already sufficient for a disclosed
reproducibility-limitation framing, consistent with the task's own guidance.

## Correction (2026-09-13)

Extending the canonical analysis to `gemma27b` exposed a real parser bug
(ODR's `force_benchmark_search` safety net was invisible to branch-point
detection — see `analysis/CHANGELOG.md`). Fixing it
changed `llama8b`'s PCR-eligible N from 19,696 to 20,753 (+5.4%) and shifted
PCR@k by a few tenths of a percentage point at every checkpoint; `qwen14b`
moved only trivially (21,408 → 21,418). Every route/PCR/direct-exit number
in THIS report and in `direct_exit_content_audit.csv`/
`route_breakdown_*.csv` was computed against the PRE-FIX qwen14b/llama8b
data and is now superseded at the percentage level (not at the qualitative-
conclusion level — the "usually direct answers" finding, the exclusion
reasoning, and the identity checks are all still valid in kind). Re-run
`direct_exit_content_audit.py`/`route_breakdown.py` against the regenerated
`canonical_run_level_completion.csv` before citing exact percentages from
this report going forward.

# RESULTS_CANONICAL_INDEX.md

Organizational control document. This file declares which packages, code
paths, and directories are canonical for the Qwen14B/Llama8B results
consolidation, and which are deprecated or superseded and must not be
cited going forward. It does not move, rename, or delete anything --
every path named below already exists exactly where it is named.

Frozen commit: `9c327466ab8e730ba29a71db375ea5aa7556981d`.

---

## 1. Canonical primary results package

**`qwen_llama_results_bundle/`**

PCR, route decomposition, final-report-input evidence, and final-answer
results for the 2-model x 4-dataset x 12-condition matrix. Contains
`MASTER_RESULTS.md` (the narrative report), `data/` (13 required CSVs +
`mechanism_run_level.csv`/`mechanism_summary.csv` + `VALIDATION_GATES.csv`),
`tables/`, `appendix/`, `figures/`, and the `.xlsx` workbook. Regenerate via
`python -m analysis.canonical.results_bundle.build_bundle --stage all`
(see `qwen_llama_results_bundle/README.md`).

## 2. Canonical local-completion evidence package

**`analysis/evidence_hop_coverage/`**

Evidence coverage measured at the researcher's own local completion step
T_i (not at final-report-generation input -- see §5 below), plus
hop/evidence-count stratification of both evidence coverage and PCR, plus
the evidence-incompleteness-vs-answer-harm association check and the
integrated recovery/evidence-boundary summary. Regenerate via
`python -m analysis.evidence_hop_coverage.build_evidence_hop_coverage`
(see `analysis/evidence_hop_coverage/README.md`).

## 3. Canonical PCR implementation

**`analysis/canonical/canonical_completion.py`**

The sole source of truth for T_i, terminal route, and PCR eligibility.
Both packages in §1 and §2 read this module (or its already-computed CSV
output) and never redefine PCR independently. See
`analysis/canonical/chapter3_final_source_of_truth.md` for the full locked
definitions and revision history (including the 2026-09-13 forced-search
branch-point fix that all current canonical numbers post-date).

## 4. Deprecated PCR implementation -- must not be used

**`analysis/pcr.py`**

Legacy dual-denominator PCR implementation, superseded by §3. Still
importable and still wired into `analysis/cli.py --stage pcr`/`--stage
statistics`. Do not cite its output, and do not extend it. Any new PCR
work must go through `analysis/canonical/canonical_completion.py`.

## 5. Evidence-boundary distinction

Two different, both-legitimate measurements of "evidence coverage" exist
in this repository, at two different points in the pipeline. They are
**not interchangeable** and will report different numbers for the same
comparison by design.

| | Local researcher completion at T_i | Final-report-generation input |
|---|---|---|
| Package | `analysis/evidence_hop_coverage/` | `qwen_llama_results_bundle/data/evidence_*.csv` |
| Method | Raw retrieved-paragraph-ID set intersection | Citation-ID extraction from `final_report_generation`'s actual input text |
| When | Upstream -- at the researcher's own first post-observation completion decision | Downstream -- after any supervisor redelegation and compression |
| Key field | `run_level_evidence_hops.csv::local_coverage_fraction` | `evidence_run_level.csv`/`evidence_summary.csv::*coverage*` (built from `EvidenceRecord.final_input_coverage_fraction`) |

The two packages' underlying per-row retrieval accounting was cross-checked
and found consistent (>99% exact per-row agreement between
`evidence_run_level.csv::local_completion_coverage` and
`run_level_evidence_hops.csv::local_coverage_fraction` wherever both are
defined -- see `analysis/evidence_hop_coverage/README.md`'s reconciliation
section). **Do not subtract the two boundaries from each other, and do not
describe a difference between them as "supervisor recovery."** Supervisor
recovery is its own, separately computed quantity (`recovery_gain` in
`run_level_evidence_hops.csv`, and `supervisor_recovery_summary.csv`), not
a byproduct of comparing these two different-question measurements.

## 6. Semantic-accuracy distinction

Three distinct things, not to be conflated:

1. **Current automatic semantic accuracy** -- fully computed today, for
   every valid answer row, by `attack/answer_scoring/answer_evaluator_v2.py::semantic_accuracy()`.
   Present in `qwen_llama_results_bundle/data/answer_run_level.csv`'s
   `automatic_semantic_label`/`exact_match`/`token_f1` columns and
   summarized in `answer_summary.csv`. This is NOT missing and should not
   be described as such.
2. **Flagged cases requiring adjudication** -- the 543 rows where the
   automatic rule itself set `needs_adjudication_flag == True` (ambiguous
   extraction, multi-candidate containment, or high token overlap without
   a safe match). See `qwen_llama_results_bundle/adjudication/` (this
   task's new output) for the blinded review packet.
3. **Future adjudicated semantic accuracy** -- does not exist yet. Requires
   a completed human review of the 543-row queue in (2), followed by
   running `qwen_llama_results_bundle/adjudication/apply_semantic_adjudication.py`
   (not run by this task -- see that script's own header). Its output will
   land in `data/answer_run_level_adjudicated.csv`,
   `answer_summary_adjudicated.csv`, `answer_paired_flips_adjudicated.csv`
   -- new files, never overwriting the pre-adjudication ones named above.

## 7. Historical/superseded paths that must not be cited

These predate the 2026-09-13 forced-search branch-point fix
(`analysis/canonical/chapter3_final_source_of_truth.md`) and/or the current
scope lock. Spot-checked against current canonical numbers and confirmed
different (see `REPOSITORY_RESULTS_INVENTORY.md` §5) -- not merely older
copies of the same numbers.

- `results/evaluation/_archive_superseded_20260910/`
- `results/evaluation/final_qwen_llama_paper_analysis_20260910/`
- `results/evaluation/v3_final_completed_qwen_llama_20260910/`
- `analysis/canonical/retrieval_equivalence_summary.md`

None of these were moved or deleted by this task. They remain on disk,
unused as a citation source.

## 8. Chapter 5 source order

1. **PCR and route results** -- `qwen_llama_results_bundle/MASTER_RESULTS.md`
   §7.1-7.5 (PCR/route content), `data/pcr_stepwise.csv`,
   `data/pcr_paired_deltas_stepwise.csv`, `data/route_decomposition_stepwise.csv`.
2. **Local-completion evidence** -- `analysis/evidence_hop_coverage/EVIDENCE_HOP_RESULTS.md`,
   `run_level_evidence_hops.csv`, `evidence_hop_summary.csv`,
   `paired_evidence_deltas.csv`, `hop_stratified_pcr_*.csv`.
3. **Final-report-input evidence** -- `qwen_llama_results_bundle/MASTER_RESULTS.md`'s
   evidence sections, `data/evidence_run_level.csv`, `data/evidence_summary.csv`.
4. **Final-answer results** -- `qwen_llama_results_bundle/data/answer_*.csv`
   (pre-adjudication), `analysis/evidence_hop_coverage/EVIDENCE_VS_ANSWER_HARM_NOTES.md`
   (preliminary association check), and, once complete,
   `qwen_llama_results_bundle/adjudication/`'s adjudicated outputs.
5. **Limitations and negative findings** -- `qwen_llama_results_bundle/MASTER_RESULTS.md`
   §7.6, and the "Known limitations" sections of both packages' `README.md`
   files.

Writing Chapter 5 in this order avoids citing a downstream number (e.g.
final-answer accuracy) before the upstream number it depends on (PCR
eligibility, local-completion evidence) has been established in the same
document.

# Table index

All tables below are generated from the canonical CSVs in `data/`, never
hand-typed. Regenerate everything with:

    python -m analysis.canonical.results_bundle.build_bundle --stage data
    python -m analysis.canonical.results_bundle.build_bundle --stage tables

## Sign conventions (read before citing any delta)

- `pcr_paired_deltas_stepwise.csv::delta_pcr` = `cue_pcr - control_pcr`.
  Positive = the attack condition completed MORE by step k (attack
  accelerated completion).
- `evidence_summary.csv::paired_evidence_loss` = `control_mean_coverage -
  cue_mean_coverage`. Positive = the cue condition lost evidence relative to
  control (spec 3.2's literal wording -- the OPPOSITE sign direction from
  `delta_pcr` above; both independently mean "positive = attack made it
  worse").

## Which "evidence coverage" boundary is this? (read before comparing across packages)

`evidence_summary.csv`'s `control_mean_coverage`/`cue_mean_coverage` are
built from `EvidenceRecord.final_input_coverage_fraction` --
**final-report-input coverage**: citation-based coverage of what
`final_report_generation` actually received, extracted from that node's own
raw input text (see `analysis/evidence.py`'s module docstring, boundary 4).
This is a **downstream** boundary, after any supervisor redelegation and
after whatever the compression/synthesis step preserved.

This is a **different, not-directly-comparable** number from
`analysis/evidence_hop_coverage/paired_evidence_deltas.csv`'s
`paired_evidence_loss`, which is built from `local_coverage_fraction` --
**local-completion coverage at T_i**: raw retrieved-paragraph-ID coverage at
the researcher's own first post-observation completion decision (the same
`T_i` this repo's PCR numbers use), computed by ID-set intersection, never
by citation-text parsing. This is an **upstream** boundary, before any
redelegation or compression.

Both are legitimate, both are "paired evidence loss," and for the same
`(model, dataset, comparison)` cell they will report **different numbers**
-- this is expected, not a bug (spec 3.2: "coverage at local completion and
final-run coverage are separate outcomes"). The two packages' underlying
per-row retrieval computation was cross-checked and found consistent (>99%
exact per-row agreement between this package's own
`evidence_run_level.csv::local_completion_coverage` and
`evidence_hop_coverage`'s `local_coverage_fraction` wherever both are
defined -- the remaining <1% is `evidence_hop_coverage` correctly reporting
`NA` for `MAX_BUDGET_EXIT` rows that this package's broader field does not
gate on route). See `analysis/evidence_hop_coverage/README.md`'s
reconciliation section for the full cross-check.

**Rule of thumb**: cite `evidence_summary.csv` for "did the final answer
generator actually see the evidence." Cite
`analysis/evidence_hop_coverage/paired_evidence_deltas.csv` for "had the
researcher retrieved the evidence by the time it decided to stop
searching." Never quote one as if it were the other.

## `route_decomposition_stepwise.csv` vs `pcr_stepwise.csv`

These two files do not repeat each other's columns. `pcr_stepwise.csv` has
`pcr`/`n_total`/`n_eligible`/route-specific PCR percentages per step.
`route_decomposition_stepwise.csv` has only the raw route-specific counts
per step plus final route tallies. Join on `(model, dataset, condition,
step)` to combine them.

## Files

- `main_chapter_tables.md` -- compact PCR@{1,5,10,30} tables (spec section 8).
- `full_pcr_tables.md` -- every PCR value, every step 1-29 plus 30, every condition.
- `full_route_tables.md` -- final terminal-route tallies for all 7 canonical routes.
- `evidence_tables.md` -- paired evidence-loss summary.
- `answer_tables.md` -- per-cell answer summary + paired answer-flip table.
- `../appendix/*.tex` -- the same underlying numbers as LaTeX booktabs tables.

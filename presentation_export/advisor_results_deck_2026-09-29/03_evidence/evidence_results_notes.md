# Evidence Results — Notes and Boundary Definition

## What "required-evidence coverage at completion" means here

The proportion of a dataset's annotated supporting evidence the researcher
had actually collected **at the moment it made its first qualifying
completion decision** (T_i — the same step PCR is measured at). This is
the **local-completion boundary**, not the final-report-generation-input
boundary used elsewhere in this repo (`qwen_llama_results_bundle/data/evidence_*.csv`'s
`final_run_coverage`/`gold_document_recall` columns measure a different,
later point — after any supervisor redelegation and compression; see
`RESULTS_CANONICAL_INDEX.md` section 5 for the full two-boundary
distinction). Mixing the two would silently combine different metric
definitions — this export does not do that.

## Source asymmetry (real, disclosed — not fabricated)

- **Qwen/Llama**: coverage-at-completion and its statistics come from
  `analysis/evidence_hop_coverage/` — a dedicated package with
  hop-stratification, hand-validated fixtures, and (in `supervisor_recovery_summary.csv`)
  a properly-labeled recovery breakdown. That package is explicitly scoped
  to `CANONICAL_BUNDLE_MODELS = (qwen14b, llama8b)` only
  (`build_evidence_hop_coverage.py:57`).
- **GPT-OSS**: no equivalent dedicated package exists. Its coverage-at-completion
  numbers in this export are computed directly from
  `gemma_gptoss_results_bundle/data/evidence_run_level.csv`'s
  `local_completion_coverage` column — the *same underlying quantity*,
  computed by the same reused `evidence_ext.py` code both bundles share —
  but the **evidence-loss and recovery statistics for GPT-OSS were computed
  fresh for this export** (a simple paired mean-difference matched by
  `(dataset, example_id)` against `NaturalClean`, and a recovery rate read
  directly off `recovery_after_local_completion`'s label values), not run
  through qwen/llama's dedicated bootstrap-CI/hop-stratified pipeline.
  Treat GPT-OSS's evidence numbers as real and comparable in *definition*,
  but less rigorously validated (no bootstrap CI, no hand-verified fixture)
  than qwen/llama's.

## EvidenceLoss sign convention

`EvidenceLoss = Coverage_Clean − Coverage_condition`. Positive = the
condition collected **less** evidence than Clean at completion. The two
Oracle rows show a **large negative** value (roughly −17 to −20pp) in every
model's table — this is mechanical, not a finding: Oracle conditions supply
100% of annotated evidence directly, so "loss" relative to Clean's real,
incomplete BM25 retrieval is definitionally negative (a coverage *gain*,
not a loss). **This is a calibration difference from an evidence-regime
change, not an attack effect** — do not read it as one.

## Headline pattern across all three models

Evidence loss vs Clean for the two strongest natural-retrieval cues (Fixed
Sufficiency / Saturation) is small in absolute terms for every model
(Qwen: ~2.3-2.4pp; Llama: see per-model table; GPT-OSS: ~1.0-1.5pp) —
real and directionally consistent (the strongest PCR-effect cues also show
the largest evidence loss, for every model), but modest in magnitude. No
model shows evidence coverage collapsing under any cue.

## Distribution figures (2026-10-06 addition)

`{qwen,llama,gpt_oss}_evidence_loss_distribution.png` show the full
per-example distribution behind each mini table's single mean
`evidence_loss_vs_clean_pp` value (violin + box, one per condition, paired
by (dataset, example_id) against Clean — identical pairing to the mini
table, so each plot's mean marker matches the table exactly). **The mean
alone understates how spread out these values are**: every natural-retrieval
condition's violin spans the full ±100pp range with visible secondary bumps
around ±50pp (consistent with discrete per-example required-evidence counts
— e.g. missing 1 of 2 required items is a 50pp swing for a 2-hop example),
even though the mean sits near zero. The two Oracle rows show most of their
mass concentrated between 0 and −50pp (a real coverage *gain*, as expected
mechanically), not a point mass at a single value. Read the mean as "typical
direction," not "typical magnitude" — the per-example spread is the more
complete picture.

## What's in the appendix (`05_appendix/evidence/`)

Full dataset-level and hop-stratified detail for Qwen/Llama
(`evidence_hop_summary.csv`, `hop_stratified_pcr_stepwise.csv`,
`hop_stratified_pcr_paired_deltas.csv`, `supervisor_recovery_summary.csv`,
`paired_evidence_deltas.csv`, `EVIDENCE_HOP_RESULTS.md`) and the full
per-condition-per-dataset GPT-OSS evidence table derived for this export.

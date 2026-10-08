# Claims and Caveats

Every number below is read directly from this deck's own generated CSVs
(`02_pcr/mini_tables/`, `03_evidence/`, `04_answers/`) — cross-check any
number by opening the named file.

## Qwen2.5-14B-Instruct

- **Largest PCR difference**: Oracle Complete–Saturation vs Clean, +30.2pp
  at PCR@1 (`02_pcr/mini_tables/cross_model_pcr_summary.csv`) — but this is
  a **calibration** comparison (evidence regime changes), not an attack
  effect. Among real attacks, Fixed Sufficiency Saturation is largest
  (+25.5pp @1).
- **Effect is immediate**: every real cue's effect for this model peaks at
  PCR@1 and has largely vanished by PCR@5 (data-derived peak checkpoint =
  @1, mean |Δ| = 14.2pp across the 9 main conditions).
- **Strongest cues**: Fixed Sufficiency (+27.2pp) and Fixed Sufficiency
  Saturation (+25.5pp) — an order of magnitude larger than single cue-only
  attacks (Epistemic Confirmation +6.9pp, Search Match +12.8pp).
- **Does source coverage strengthen the primary cue?** No — Epistemic
  Confirmation alone (+6.9pp) vs + Source Coverage (+7.1pp), Search Match
  alone (+12.8pp) vs + Source Coverage (+11.9pp): both essentially
  unchanged, one even slightly smaller. No stacking effect.
- **Evidence coverage decreases**: yes, modestly — 1.7-2.4pp loss vs Clean
  for the four strongest cues (`03_evidence/qwen_evidence_mini_table.csv`),
  growing with cue strength but never large.
- **Final-answer harm**: no consistent effect. Net paired harm ranges
  −0.4pp to +0.7pp across all 9 conditions — small and inconsistent in
  sign, not a real effect.
- **Adjudication**: pre-adjudication throughout (see
  `04_answers/ANSWER_ACCURACY_STATUS.md`).

## Llama-3.1-8B-Instruct

- **Largest PCR difference**: Oracle Complete–Saturation vs Clean, +17.6pp
  at PCR@5 — calibration, not attack. Among real attacks, Fixed Sufficiency
  Saturation is largest (+16.5pp @5).
- **Effect is delayed relative to Qwen**: this model's data-derived peak
  checkpoint is @5 (mean |Δ| = 10.9pp), not @1 — the same cues that move
  Qwen immediately take one additional decision cycle to show up for Llama.
  Reporting only a PCR@1 comparison across models would understate Llama's
  real effect size.
- **Strongest cues**: Fixed Sufficiency (+15.4pp) and Fixed Sufficiency
  Saturation (+16.5pp) — again far larger than single cues (Epistemic
  Confirmation +8.7pp, Search Match +8.1pp).
- **Does source coverage strengthen the primary cue?** Mixed/small: +
  Source on Epistemic Confirmation raises it from +8.7pp to +11.2pp; +
  Source on Search Match raises it from +8.1pp to +13.2pp. Real but modest
  — unlike Qwen, where stacking showed no effect at all. This is a genuine
  candidate cross-model difference worth flagging as suggestive, not
  confirmed (see `qwen_llama_results_bundle/CUE_EFFECT_ANALYSIS.md` §3 for
  the full paired-CI analysis behind this).
- **Evidence coverage decreases**: yes, slightly larger than Qwen's — up to
  3.0pp loss for Fixed Sufficiency Saturation.
- **Final-answer harm**: no consistent effect (−0.6pp to +0.9pp across
  conditions, inconsistent sign).
- **Adjudication**: pre-adjudication, same shared queue as Qwen.

## GPT-OSS-20B

- **Scope reminder**: 3 of 4 datasets only (HotpotQA, MuSiQue, MultiHopRAG
  — 2WikiMultiHopQA is entirely absent, rerun queued but not started at
  export time).
- **Largest PCR difference**: Oracle Complete–Saturation vs Clean, +16.0pp
  at PCR@5 — calibration, not attack. Among real attacks, Fixed Sufficiency
  Saturation is largest (+12.7pp @5).
- **Effect timing**: data-derived peak checkpoint is @5 (mean |Δ| = 7.5pp)
  — same timing pattern as Llama, not Qwen. **Do not assume GPT-OSS behaves
  like either existing model** — this was checked from its own curves, not
  inherited from the Qwen/Llama narrative.
- **Strongest cues**: Fixed Sufficiency (+10.8pp) and Fixed Sufficiency
  Saturation (+12.7pp) — the same qualitative ranking as both other models
  (sufficiency claims >> single cues), though every effect size for
  GPT-OSS is noticeably smaller in absolute terms than the corresponding
  Qwen/Llama number (e.g. Epistemic Confirmation: GPT-OSS +1.4pp vs Qwen
  +6.9pp vs Llama +8.7pp).
- **Does source coverage strengthen the primary cue?** Yes, and more
  visibly than either other model: Epistemic Confirmation alone (+1.4pp)
  vs + Source (+4.9pp); Search Match alone (+6.7pp) vs + Source (+7.7pp).
  **This is GPT-OSS's own actual result — it is not forced into the
  Qwen "no stacking" or Llama "modest stacking" narrative.**
- **Evidence coverage decreases**: smaller than either other model in
  absolute terms (0.4-1.5pp for the four strongest natural-retrieval cues,
  vs Qwen's 1.7-2.4pp and Llama's 1.4-3.0pp); two conditions
  (Epistemic Confirmation, Epistemic Confirmation+Source) show a small
  *negative* loss (i.e., a small coverage gain, −0.5pp) — noise around
  zero, not a real effect.
- **Final-answer harm**: no consistent effect (−0.8pp to +1.9pp, inconsistent
  sign) — same pattern as both other models.
- **Adjudication**: pre-adjudication; no adjudication queue has been built
  for this model yet (a real, separate gap from Qwen/Llama's in-progress
  queue — see `04_answers/ANSWER_ACCURACY_STATUS.md`).

## Cross-cutting, all three models

- **The sufficiency-claim ranking replicates in all three models**: Fixed
  Sufficiency and Fixed Sufficiency Saturation are the two largest
  real-attack effects for every model, at every model's own peak
  checkpoint. This is the single most robust finding in the whole study.
- **A timing split exists, and GPT-OSS sides with Llama, not Qwen**: Qwen's
  effect peaks at PCR@1; both Llama and GPT-OSS peak at PCR@5. This was
  determined from GPT-OSS's actual curves (`02_pcr/gpt_oss/`), not assumed
  from the other two models' pattern.
- **Final-answer accuracy is never reliably affected, for any model, by any
  cue** — the most consistent negative finding across the whole deck.
- **Dataset limitation**: GPT-OSS's numbers above are macro-averaged over 3
  datasets, not 4 — direct magnitude comparisons to Qwen/Llama (4-dataset
  averages) should be read as approximate, not exact, until 2WikiMultiHopQA
  completes.
- **Denominator**: every number above uses `pcr_eligible` (PCR),
  `evidence_eligible` (evidence), or the paired-flip denominator
  (answers) — never raw row counts. Eligible-N is in every mini table.
- **Semantic-adjudication status**: pre-adjudication for all three models,
  no exceptions, anywhere in this deck.

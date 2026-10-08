# Slide Asset Index

Exact file(s) to use for each planned slide. All paths relative to this
ZIP's root.

## 1. Title
No data asset — title/authorship only.

## 2. Research question
No data asset — narrative slide. Ground it in `00_READ_ME_FIRST.md`'s
framing and `CLAIMS_AND_CAVEATS.md`'s opening.

## 3. Attack point (where in the pipeline the cue is injected)
`01_conditions/slide_condition_table.png` (visual overview) — describe
verbally that the cue is appended to the retrieved passage text the
researcher subgraph observes, before its first post-observation decision.

## 4. 12 experimental conditions
`01_conditions/slide_condition_table.png` (slide image, shortened text) +
hand off `01_conditions/all_12_conditions.csv` / `.md` if the advisor wants
the verbatim cue text.

## 5. Evaluation setup (models, datasets, denominators)
Build from `00_READ_ME_FIRST.md`'s model/dataset table +
`02_pcr/PCR_DEFINITION.md`'s denominator section.

## 6. PCR definition
`02_pcr/PCR_DEFINITION.md` (the whole file fits one or two slides — locked
definition, denominator, terminal routes, validation table).

## 7. Qwen PCR curves and mini table
`02_pcr/qwen/pcr_qwen_cue_only.png`, `pcr_qwen_source_attached.png`,
`pcr_qwen_fixed_sufficiency.png`, `pcr_qwen_oracle_calibration.png` (pick
1-4 depending on slide budget) + `02_pcr/mini_tables/qwen_pcr_mini_table.png`.

## 8. Llama PCR curves and mini table
`02_pcr/llama/pcr_llama_cue_only.png`, `pcr_llama_source_attached.png`,
`pcr_llama_fixed_sufficiency.png`, `pcr_llama_oracle_calibration.png` +
`02_pcr/mini_tables/llama_pcr_mini_table.png`.

## 9. GPT-OSS PCR curves and mini table
`02_pcr/gpt_oss/pcr_gpt_oss_cue_only.png`, `pcr_gpt_oss_source_attached.png`,
`pcr_gpt_oss_fixed_sufficiency.png`, `pcr_gpt_oss_oracle_calibration.png` +
`02_pcr/mini_tables/gpt_oss_pcr_mini_table.png`. **State verbally that
2WikiMultiHopQA is absent from every one of these.**

## 10. Cross-model PCR comparison
`02_pcr/mini_tables/cross_model_pcr_summary.png` (each model's own
data-derived peak checkpoint — Qwen@1, Llama@5, GPT-OSS@5 — is stated in
the table itself, not assumed).

## 11. Required-evidence coverage at completion
`03_evidence/cross_model_evidence_summary.png` for the headline comparison;
`03_evidence/{qwen,llama,gpt_oss}_evidence_mini_table.png` for per-model
detail if there's room. Verbally flag the Oracle rows as a calibration
difference, not an attack effect (large negative "loss" = evidence GAIN,
mechanical).

## 12. Final-answer accuracy
`04_answers/cross_model_answer_summary.png` for the headline comparison;
per-model mini tables if there's room. **State verbally that every number
is pre-adjudication** (`04_answers/ANSWER_ACCURACY_STATUS.md` has the exact
wording if needed).

## 13. Main findings and remaining work
Build from `CLAIMS_AND_CAVEATS.md` directly — it is written slide-ready,
one bullet list per model plus a cross-cutting section.

---

## Everything NOT listed above (appendix-only, have ready if asked)

- `05_appendix/pcr/` — full per-dataset stepwise curves/tables, route
  decomposition, run inventory, validation gates.
- `05_appendix/evidence/` — hop-stratified detail, supervisor recovery,
  paired evidence deltas (Qwen/Llama only).
- `05_appendix/answers/` — full per-dataset (non-macro-averaged) answer
  tables.

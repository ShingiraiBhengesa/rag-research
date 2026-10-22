# E1 sensitivity checks

Generated from the published bundles (`qwen_llama/`, `gemma_gptoss/`) by:

    python -m prematurestop.analysis.fixed_cue_matrix.sensitivity_checks

## Identical-d1 sensitivity

The methods chapter commits to re-estimating the cue effects on the strict
subset of pairs whose pre-cue first search observation (d₁) is byte-identical
(`identical_base_tool_message_hash` in `data/retrieval_equivalence.csv`).

| File | Contents |
|---|---|
| `identical_d1_pcr_deltas.csv` | Paired ΔPCR@k, k = 1–30, for all 14 comparisons in the retrieval audit × 4 models × 4 datasets, on `all_paired` and on `identical_d1` |
| `identical_d1_summary.md` | Fixed Sufficiency and Suff. + Saturation vs Neutral at each model's characteristic step (Qwen 1, Llama 3, Gemma 1, gpt-oss 2) |

**Method.** The pairing rule, T_i, the bootstrap (2,000 resamples, one pass
across steps) and the seed are the same as for `data/pcr_paired_deltas_stepwise.csv`.
The script asserts that its `all_paired` rows reproduce that published file
(same n, same Δ).

**Result.**
- **Sign:** restricted to identical-d₁ pairs, the cue effect keeps its sign in
  **30 of 32** model × dataset × cue cells.
- **Significance:** the identical-d₁ CI still excludes 0 in 25 of 32 cells.
- **Where it fails:** in gpt-oss cells with only 16–28 identical pairs, and in
  one Gemma MultiHop-RAG cell (n = 47).
- **Sign flips:** the two sign flips are gpt-oss MuSiQue (n = 27 and n = 25),
  whose CIs include 0.

**Subset size.** Identical-d₁ pairs are 3.6–43.1% of pairs, depending on model
and dataset. This is much smaller than the 18–95% of pairs with an identical
retrieved document *set*. The two are different quantities: the d₁ text also
depends on the agent's exact query.

## Technical failures by condition

| File | Contents |
|---|---|
| `technical_failures_by_condition.csv` | `n_failures / n_completed` per model × dataset × condition (from `data/run_inventory.csv`) |
| `technical_failure_condition_tests.csv` | Chi-square test of failure × condition per model × dataset and pooled per model, with the minimum expected count and a Cochran's-rule validity flag |

**Results.**
- **gpt-oss** (965 failures; 2Wiki has none): p = 0.42 (HotpotQA), 0.92
  (MuSiQue), 0.58 (MultiHop-RAG), 0.93 (pooled).
- **Llama** (629 failures): p ≥ 0.89 everywhere.
- **Qwen** (24 failures): too few for a chi-square test (`chi2_valid = False`),
  so its p-values should not be cited.

Failures are not concentrated in any condition, so they are unlikely to bias the
condition contrasts. The breakdown by error type (948 Harmony parser errors, 14
length errors, 3 JSON errors) comes from the local run JSONL files and is not
published here.

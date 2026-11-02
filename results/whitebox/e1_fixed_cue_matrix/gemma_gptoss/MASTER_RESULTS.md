# MASTER_RESULTS.md

Every number below cites the `data/*.csv` file it was read from -- see `tables/TABLE_INDEX.md` for sign conventions before reading any delta.

## Locked metric definitions

PCR means **Post-observation Completion Rate**, never Premature Completion Rate. For each PCR-eligible run, T_i is the first post-observation researcher decision that ends the targeted episode through RESEARCH_COMPLETE or NO_TOOL_DIRECT_EXIT. PCR@k is the cumulative fraction of eligible runs with T_i <= k; one researcher-node decision (including think_tool) is one step; the full curve uses steps 1-29, PCR@30 is the end-of-budget summary. The raw DIRECT_ANSWER implementation label is displayed as no-tool exit throughout this bundle, never as proof of a concise answer. [table: results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md]

Evidence deficit = 1 - coverage. Paired evidence loss = control coverage minus cue coverage (`data/evidence_summary.csv::paired_evidence_loss`) -- note this is the *opposite* sign direction from `data/pcr_paired_deltas_stepwise.csv::delta_pcr` (cue minus control); both independently mean "positive = attack made it worse." See [table: tables/TABLE_INDEX.md].

**This bundle's evidence-loss numbers measure final-report-input coverage** (citation-based, what `final_report_generation` actually received) -- a different, downstream boundary from `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage`'s local-completion-at-T_i evidence loss. The two are not interchangeable and will report different numbers for the same comparison by design (coverage at local completion and at final-run termination are separate outcomes); see `tables/TABLE_INDEX.md` for the full boundary disambiguation and a per-row consistency cross-check between the two packages.

Main downstream answer measure is semantic Answer Accuracy with adjudication status visible (`needs_adjudication_flag`); automatic semantic labels are not adjudicated accuracy until the blinded review is complete (spec 3.3) [table: data/answer_run_level.csv].

## 7.1 Data freeze and validity

- Git commit: `bca0b16cf8cf17f4e6ed1197f727351d79a99e79` [table: analysis_manifest.json]
- Production root: `local_data` [table: analysis_manifest.json]
- Models included: gemma27b, gpt_oss_20b [table: analysis_manifest.json]
- Active SLURM jobs for included models at freeze time: 0 [table: analysis_manifest.json] -- must be 0; if not, the bundle is not frozen.

### Planned / completed / valid / eligible counts by cell

See `data/run_inventory.csv` for the full 96-cell (2 models x 4 datasets x 12 conditions) breakdown [table: data/run_inventory.csv]. Totals:

- Planned: 43200 [table: data/run_inventory.csv]
- Completed: 42926 [table: data/run_inventory.csv]
- Valid: 41961 [table: data/run_inventory.csv]
- PCR-eligible: 41927 [table: data/run_inventory.csv]
- Evidence-eligible: 41961 [table: data/run_inventory.csv]
- Answer-eligible: 41961 [table: data/run_inventory.csv]
- Failures (technical/parser/logging): 965 [table: data/run_inventory.csv]

Metric/code version: `canonical_completion.py` (authoritative PCR/route source, see `results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md`), git commit above.

### Scope notes

- `gemma27b`: Served with the custom `gemma3_tool_code` parser (`src/prematurestop/system/tool_parsers/gemma3_tool_code.py`) for every dataset; an earlier run with the wrong `functiongemma` parser is quarantined and excluded [table: results/whitebox/e1_fixed_cue_matrix/canonical/gemma27b_results_synthesis.md].
- `gpt_oss_20b`: **Parser difference across datasets.** gpt_oss_20b/2wiki was served with the lenient Harmony tool-call parser plugin (`src/prematurestop/system/tool_parsers/gptoss_lenient_harmony.py`, submitted 2026-09-28, runs 2026-10-03/04); hotpot, musique and multihoprag were served without it (2026-09-14/15). All 12 conditions within a dataset share one parser, so within-dataset paired contrasts are valid; cross-dataset comparisons for gpt_oss_20b mix two serving configurations, and its technical-failure rates are not comparable across datasets (per-cell failures: [table: data/run_inventory.csv]).

## 7.2 Gemma-3-27B-IT results

### Compact PCR@k tables (all 12 conditions) [table: data/pcr_stepwise.csv]

## Gemma-3-27B-IT
### HotpotQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 450 | 26.4 | 100.0 | 100.0 | 100.0 |
| Neutral | 450 | 30.7 | 100.0 | 100.0 | 100.0 |
| ContinueSearch | 450 | 13.8 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation | 450 | 39.1 | 100.0 | 100.0 | 100.0 |
| Search-match cue | 450 | 36.2 | 100.0 | 100.0 | 100.0 |
| Source coverage | 450 | 30.9 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation + source | 450 | 36.9 | 100.0 | 100.0 | 100.0 |
| Search-match + source | 450 | 40.2 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiency | 450 | 48.2 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiencySaturation | 450 | 50.4 | 100.0 | 100.0 | 100.0 |
| Oracle Neutral | 450 | 46.0 | 100.0 | 100.0 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 450 | 78.2 | 100.0 | 100.0 | 100.0 |

### 2WikiMultiHopQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 450 | 20.2 | 100.0 | 100.0 | 100.0 |
| Neutral | 450 | 20.2 | 100.0 | 100.0 | 100.0 |
| ContinueSearch | 450 | 15.1 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation | 450 | 31.1 | 100.0 | 100.0 | 100.0 |
| Search-match cue | 450 | 30.9 | 100.0 | 100.0 | 100.0 |
| Source coverage | 450 | 24.4 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation + source | 449 | 31.4 | 100.0 | 100.0 | 100.0 |
| Search-match + source | 450 | 30.4 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiency | 450 | 40.2 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiencySaturation | 350 | 36.0 | 100.0 | 100.0 | 100.0 |
| Oracle Neutral | 450 | 56.4 | 100.0 | 100.0 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 450 | 82.4 | 100.0 | 100.0 | 100.0 |

### MuSiQue

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 449 | 14.7 | 100.0 | 100.0 | 100.0 |
| Neutral | 400 | 16.2 | 100.0 | 100.0 | 100.0 |
| ContinueSearch | 450 | 11.3 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation | 450 | 22.7 | 100.0 | 100.0 | 100.0 |
| Search-match cue | 450 | 21.3 | 100.0 | 100.0 | 100.0 |
| Source coverage | 450 | 16.2 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation + source | 450 | 19.1 | 100.0 | 100.0 | 100.0 |
| Search-match + source | 450 | 18.0 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiency | 450 | 26.4 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiencySaturation | 450 | 32.4 | 100.0 | 100.0 | 100.0 |
| Oracle Neutral | 450 | 22.2 | 100.0 | 100.0 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 450 | 53.6 | 100.0 | 100.0 | 100.0 |

### MultiHopRAG

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 450 | 27.3 | 100.0 | 100.0 | 100.0 |
| Neutral | 450 | 25.8 | 100.0 | 100.0 | 100.0 |
| ContinueSearch | 450 | 15.8 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation | 450 | 28.9 | 100.0 | 100.0 | 100.0 |
| Search-match cue | 450 | 26.7 | 100.0 | 100.0 | 100.0 |
| Source coverage | 450 | 26.0 | 100.0 | 100.0 | 100.0 |
| Epistemic confirmation + source | 450 | 30.2 | 100.0 | 100.0 | 100.0 |
| Search-match + source | 450 | 26.9 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiency | 450 | 52.0 | 100.0 | 100.0 | 100.0 |
| PreRegisteredSufficiencySaturation | 450 | 49.3 | 100.0 | 100.0 | 100.0 |
| Oracle Neutral | 450 | 29.8 | 100.0 | 100.0 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 450 | 54.7 | 100.0 | 100.0 | 100.0 |

### Paired PCR deltas at PCR@{1,5,10,30}, all comparisons [table: data/pcr_paired_deltas_stepwise.csv]

| Dataset | Comparison | n_paired | ΔPCR@1 | ΔPCR@5 | ΔPCR@10 | ΔPCR@30 |
|---|---|---|---|---|---|---|
| 2WikiMultiHopQA | Continue Search vs Natural Neutral | 450 | -5.1% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 449 | 0.2% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Epistemic Confirmation vs Natural Neutral | 450 | 10.9% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Fixed Sufficiency Saturation vs Natural Neutral | 350 | 14.6% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Fixed Sufficiency vs Natural Neutral | 450 | 20.0% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Natural Neutral vs Natural Clean | 450 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 450 | 20.0% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 350 | 14.9% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 450 | 10.9% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 450 | 10.7% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Oracle Complete-Saturation vs Oracle Complete-Neutral | 450 | 26.0% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Search Match + Source Coverage vs Search Match | 450 | -0.4% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Search Match vs Natural Neutral | 450 | 10.7% | 0.0% | 0.0% | 0.0% |
| 2WikiMultiHopQA | Source Coverage vs Natural Neutral | 450 | 4.2% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Continue Search vs Natural Neutral | 450 | -16.9% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 450 | -2.2% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Epistemic Confirmation vs Natural Neutral | 450 | 8.4% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Fixed Sufficiency Saturation vs Natural Neutral | 450 | 19.8% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Fixed Sufficiency vs Natural Neutral | 450 | 17.6% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Natural Neutral vs Natural Clean | 450 | 4.2% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 450 | 21.8% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 450 | 24.0% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 450 | 12.7% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 450 | 9.8% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Oracle Complete-Saturation vs Oracle Complete-Neutral | 450 | 32.2% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Search Match + Source Coverage vs Search Match | 450 | 4.0% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Search Match vs Natural Neutral | 450 | 5.6% | 0.0% | 0.0% | 0.0% |
| HotpotQA | Source Coverage vs Natural Neutral | 450 | 0.2% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Continue Search vs Natural Neutral | 450 | -10.0% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 450 | 1.3% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Epistemic Confirmation vs Natural Neutral | 450 | 3.1% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Fixed Sufficiency Saturation vs Natural Neutral | 450 | 23.6% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Fixed Sufficiency vs Natural Neutral | 450 | 26.2% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Natural Neutral vs Natural Clean | 450 | -1.6% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 450 | 24.7% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 450 | 22.0% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 450 | 1.6% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 450 | -0.7% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Oracle Complete-Saturation vs Oracle Complete-Neutral | 450 | 24.9% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Search Match + Source Coverage vs Search Match | 450 | 0.2% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Search Match vs Natural Neutral | 450 | 0.9% | 0.0% | 0.0% | 0.0% |
| MultiHopRAG | Source Coverage vs Natural Neutral | 450 | 0.2% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Continue Search vs Natural Neutral | 400 | -4.5% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 450 | -3.6% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Epistemic Confirmation vs Natural Neutral | 400 | 7.2% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Fixed Sufficiency Saturation vs Natural Neutral | 400 | 17.5% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Fixed Sufficiency vs Natural Neutral | 400 | 11.8% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Natural Neutral vs Natural Clean | 399 | 1.5% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 449 | 11.8% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 449 | 17.8% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 449 | 8.0% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 449 | 6.7% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Oracle Complete-Saturation vs Oracle Complete-Neutral | 450 | 31.3% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Search Match + Source Coverage vs Search Match | 450 | -3.3% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Search Match vs Natural Neutral | 400 | 5.0% | 0.0% | 0.0% | 0.0% |
| MuSiQue | Source Coverage vs Natural Neutral | 400 | 0.5% | 0.0% | 0.0% | 0.0% |

### Full stepwise PCR (every step 1-30, all 12 conditions) [table: data/pcr_stepwise.csv]

Values in %. Column codes:

| Code | Condition |
|---|---|
| Clean | Clean |
| Neutral | Neutral |
| ContSearch | ContinueSearch |
| EpistConf | Epistemic confirmation |
| SearchMatch | Search-match cue |
| SrcCov | Source coverage |
| EpistConf+Src | Epistemic confirmation + source |
| SearchMatch+Src | Search-match + source |
| PreRegSuff | PreRegisteredSufficiency |
| PreRegSuffSat | PreRegisteredSufficiencySaturation |
| OracleNeutral | Oracle Neutral |
| OraclePreRegSuffSat | Oracle PreRegisteredSufficiencySaturation |

## Gemma-3-27B-IT / HotpotQA
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |    26.4 |         13.8 |      30.7 |         48.2 |            50.4 |        39.1 |            36.9 |          36.2 |              40.2 |     30.9 |            46   |                  78.2 |
|      2 |    93.3 |         95.1 |      95.8 |         97.3 |            97.6 |        96.4 |            96   |          96   |              96.7 |     94.9 |            94.2 |                  98.9 |
|      3 |    99.8 |         99.6 |      99.8 |         99.8 |            99.8 |       100   |           100   |          99.8 |              99.6 |    100   |           100   |                 100   |
|      4 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      5 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      6 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      7 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      8 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      9 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     10 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     11 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     12 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     13 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     14 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     15 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     16 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     17 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     18 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     19 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     20 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     21 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     22 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     23 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     24 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     25 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     26 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     27 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     28 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     29 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     30 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |

## Gemma-3-27B-IT / 2WikiMultiHopQA
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |    20.2 |         15.1 |      20.2 |         40.2 |            36   |        31.1 |            31.4 |          30.9 |              30.4 |     24.4 |            56.4 |                  82.4 |
|      2 |    95.3 |         94   |      92.7 |         97.3 |            97.4 |        96   |            94.9 |          96   |              95.8 |     94.9 |            94   |                  98.7 |
|      3 |   100   |        100   |     100   |        100   |           100   |        99.8 |           100   |          99.6 |             100   |    100   |           100   |                  99.8 |
|      4 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |          99.8 |             100   |    100   |           100   |                 100   |
|      5 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      6 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      7 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      8 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      9 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     10 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     11 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     12 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     13 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     14 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     15 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     16 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     17 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     18 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     19 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     20 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     21 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     22 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     23 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     24 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     25 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     26 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     27 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     28 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     29 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     30 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |

## Gemma-3-27B-IT / MuSiQue
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |    14.7 |         11.3 |      16.2 |         26.4 |            32.4 |        22.7 |            19.1 |          21.3 |              18   |     16.2 |            22.2 |                  53.6 |
|      2 |    90.6 |         95.1 |      94.8 |         94.2 |            97.8 |        94.2 |            93.3 |          93.6 |              93.1 |     93.6 |            92.2 |                  96.9 |
|      3 |    99.8 |        100   |     100   |         99.8 |            99.8 |        99.8 |            99.3 |          99.6 |             100   |    100   |           100   |                 100   |
|      4 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      5 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      6 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      7 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      8 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      9 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     10 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     11 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     12 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     13 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     14 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     15 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     16 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     17 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     18 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     19 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     20 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     21 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     22 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     23 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     24 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     25 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     26 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     27 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     28 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     29 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     30 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |

## Gemma-3-27B-IT / MultiHopRAG
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |    27.3 |         15.8 |      25.8 |         52   |            49.3 |        28.9 |            30.2 |          26.7 |              26.9 |     26   |            29.8 |                  54.7 |
|      2 |    86.4 |         92.7 |      87.3 |         93.6 |            94.2 |        89.6 |            90   |          86.7 |              86.9 |     85.8 |            88   |                  95.1 |
|      3 |    99.3 |         99.3 |      98.9 |         99.8 |            99.8 |        99.8 |            99.3 |          98.4 |              98.7 |     99.1 |            98.9 |                  99.6 |
|      4 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      5 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      6 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      7 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      8 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|      9 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     10 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     11 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     12 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     13 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     14 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     15 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     16 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     17 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     18 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     19 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     20 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     21 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     22 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     23 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     24 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     25 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     26 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     27 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     28 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     29 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |
|     30 |   100   |        100   |     100   |        100   |           100   |       100   |           100   |         100   |             100   |    100   |           100   |                 100   |

### Route decomposition -- final terminal-route tallies, all 7 canonical routes [table: data/route_decomposition_stepwise.csv]

| model          | dataset         | condition                                 |   final_n_RESEARCH_COMPLETE |   final_n_NO_TOOL_DIRECT_EXIT |   final_n_MAX_BUDGET_EXIT |   final_n_TOOL_OR_PARSER_FAILURE |   final_n_TRACE_LOGGING_FAILURE |   final_n_EXECUTION_FAILURE |   final_n_OTHER_VALID_ROUTE |
|:---------------|:----------------|:------------------------------------------|----------------------------:|------------------------------:|--------------------------:|---------------------------------:|--------------------------------:|----------------------------:|----------------------------:|
| Gemma-3-27B-IT | 2WikiMultiHopQA | Clean                                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | ContinueSearch                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Neutral                                   |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | PreRegisteredSufficiency                  |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | PreRegisteredSufficiencySaturation        |                           0 |                           350 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Epistemic confirmation                    |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Epistemic confirmation + source           |                           0 |                           449 |                         0 |                                0 |                               1 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Search-match cue                          |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Search-match + source                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Source coverage                           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Oracle Neutral                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | 2WikiMultiHopQA | Oracle PreRegisteredSufficiencySaturation |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Clean                                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | ContinueSearch                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Neutral                                   |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | PreRegisteredSufficiency                  |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | PreRegisteredSufficiencySaturation        |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Epistemic confirmation                    |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Epistemic confirmation + source           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Search-match cue                          |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Search-match + source                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Source coverage                           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Oracle Neutral                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | HotpotQA        | Oracle PreRegisteredSufficiencySaturation |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Clean                                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | ContinueSearch                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Neutral                                   |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | PreRegisteredSufficiency                  |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | PreRegisteredSufficiencySaturation        |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Epistemic confirmation                    |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Epistemic confirmation + source           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Search-match cue                          |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Search-match + source                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Source coverage                           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Oracle Neutral                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MultiHopRAG     | Oracle PreRegisteredSufficiencySaturation |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Clean                                     |                           0 |                           449 |                         0 |                                0 |                               1 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | ContinueSearch                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Neutral                                   |                           0 |                           400 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | PreRegisteredSufficiency                  |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | PreRegisteredSufficiencySaturation        |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Epistemic confirmation                    |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Epistemic confirmation + source           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Search-match cue                          |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Search-match + source                     |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Source coverage                           |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Oracle Neutral                            |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |
| Gemma-3-27B-IT | MuSiQue         | Oracle PreRegisteredSufficiencySaturation |                           0 |                           450 |                         0 |                                0 |                               0 |                           0 |                           0 |

### Evidence coverage and paired evidence loss (final-report-input boundary) [table: data/evidence_summary.csv]

This is coverage of what `final_report_generation` actually received (citation-based, downstream of any redelegation/compression) -- a DIFFERENT boundary from `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage`'s local-completion-at-T_i paired evidence loss (see `tables/TABLE_INDEX.md`).

| model    | dataset     | comparison_name                                                                       | control_condition              | cue_condition                                     |   n_paired |   control_mean_coverage |   cue_mean_coverage |   paired_evidence_loss |     ci_lower |    ci_upper |
|:---------|:------------|:--------------------------------------------------------------------------------------|:-------------------------------|:--------------------------------------------------|-----------:|------------------------:|--------------------:|-----------------------:|-------------:|------------:|
| gemma27b | 2wiki       | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        450 |                0.879444 |            0.870556 |            0.00888889  | -0.00833333  |  0.0261111  |
| gemma27b | 2wiki       | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        450 |                0.870556 |            0.847778 |            0.0227778   |  0.00277778  |  0.0427778  |
| gemma27b | 2wiki       | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        450 |                0.870556 |            0.870556 |            0           | -0.0188889   |  0.0194444  |
| gemma27b | 2wiki       | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        450 |                0.870556 |            0.859444 |            0.0111111   | -0.0105556   |  0.0327778  |
| gemma27b | 2wiki       | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        350 |                0.875714 |            0.863571 |            0.0121429   | -0.00928571  |  0.0342857  |
| gemma27b | 2wiki       | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        450 |                0.879444 |            0.847778 |            0.0316667   |  0.0116667   |  0.0522222  |
| gemma27b | 2wiki       | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        450 |                0.879444 |            0.870556 |            0.00888889  | -0.01        |  0.0288889  |
| gemma27b | 2wiki       | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        450 |                0.879444 |            0.859444 |            0.02        | -0.00111111  |  0.0405556  |
| gemma27b | 2wiki       | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        350 |                0.876429 |            0.863571 |            0.0128571   | -0.00785714  |  0.0342857  |
| gemma27b | 2wiki       | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        450 |                0.870556 |            0.869444 |            0.00111111  | -0.0177778   |  0.0205556  |
| gemma27b | 2wiki       | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        450 |                1        |            1        |            0           |  0           |  0          |
| gemma27b | 2wiki       | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        450 |                0.870556 |            0.876667 |           -0.00611111  | -0.0261111   |  0.0133333  |
| gemma27b | 2wiki       | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        449 |                0.848552 |            0.861915 |           -0.013363    | -0.0334076   |  0.00723831 |
| gemma27b | 2wiki       | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        450 |                0.870556 |            0.884444 |           -0.0138889   | -0.0333333   |  0.00444444 |
| gemma27b | hotpot      | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        450 |                0.852222 |            0.836667 |            0.0155556   | -0.00444444  |  0.0355556  |
| gemma27b | hotpot      | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        450 |                0.836667 |            0.842222 |           -0.00555556  | -0.0255556   |  0.0155556  |
| gemma27b | hotpot      | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        450 |                0.836667 |            0.828889 |            0.00777778  | -0.0133333   |  0.0288889  |
| gemma27b | hotpot      | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        450 |                0.836667 |            0.813333 |            0.0233333   |  0           |  0.0466667  |
| gemma27b | hotpot      | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        450 |                0.836667 |            0.825556 |            0.0111111   | -0.0111111   |  0.0333333  |
| gemma27b | hotpot      | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        450 |                0.852222 |            0.842222 |            0.01        | -0.0111111   |  0.0311111  |
| gemma27b | hotpot      | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        450 |                0.852222 |            0.828889 |            0.0233333   |  0.00222222  |  0.0444444  |
| gemma27b | hotpot      | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        450 |                0.852222 |            0.813333 |            0.0388889   |  0.0166667   |  0.0622222  |
| gemma27b | hotpot      | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        450 |                0.852222 |            0.825556 |            0.0266667   |  0.00333333  |  0.0488889  |
| gemma27b | hotpot      | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        450 |                0.836667 |            0.848889 |           -0.0122222   | -0.0333333   |  0.00888889 |
| gemma27b | hotpot      | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        450 |                1        |            1        |            0           |  0           |  0          |
| gemma27b | hotpot      | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        450 |                0.836667 |            0.832222 |            0.00444444  | -0.0155556   |  0.0244444  |
| gemma27b | hotpot      | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        450 |                0.842222 |            0.831111 |            0.0111111   | -0.00888889  |  0.03       |
| gemma27b | hotpot      | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        450 |                0.828889 |            0.851111 |           -0.0222222   | -0.0433333   | -0.00222222 |
| gemma27b | multihoprag | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        450 |                0.978333 |            0.980556 |           -0.00222222  | -0.00777778  |  0.00333333 |
| gemma27b | multihoprag | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        450 |                0.980556 |            0.978889 |            0.00166667  | -0.00277778  |  0.00666667 |
| gemma27b | multihoprag | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        450 |                0.980556 |            0.977778 |            0.00277778  | -0.00222222  |  0.00777778 |
| gemma27b | multihoprag | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        450 |                0.980556 |            0.975    |            0.00555556  | -0.000555556 |  0.0116667  |
| gemma27b | multihoprag | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        450 |                0.980556 |            0.971667 |            0.00888889  |  0.00333333  |  0.0144444  |
| gemma27b | multihoprag | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        450 |                0.978333 |            0.978889 |           -0.000555556 | -0.00666667  |  0.00611111 |
| gemma27b | multihoprag | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        450 |                0.978333 |            0.977778 |            0.000555556 | -0.00444444  |  0.00555556 |
| gemma27b | multihoprag | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        450 |                0.978333 |            0.975    |            0.00333333  | -0.00222222  |  0.00888889 |
| gemma27b | multihoprag | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        450 |                0.978333 |            0.971667 |            0.00666667  |  0.00166667  |  0.0122222  |
| gemma27b | multihoprag | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        450 |                0.980556 |            0.979444 |            0.00111111  | -0.00444444  |  0.00666667 |
| gemma27b | multihoprag | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        450 |                1        |            1        |            0           |  0           |  0          |
| gemma27b | multihoprag | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        450 |                0.980556 |            0.978889 |            0.00166667  | -0.00388889  |  0.00722222 |
| gemma27b | multihoprag | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        450 |                0.978889 |            0.978889 |            0           | -0.005       |  0.005      |
| gemma27b | multihoprag | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        450 |                0.977778 |            0.98     |           -0.00222222  | -0.00833333  |  0.00388889 |
| gemma27b | musique     | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        399 |                0.650376 |            0.649749 |            0.000626566 | -0.0233918   |  0.0252715  |
| gemma27b | musique     | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        400 |                0.648125 |            0.649375 |           -0.00125     | -0.028125    |  0.025625   |
| gemma27b | musique     | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        400 |                0.648125 |            0.65125  |           -0.003125    | -0.0302135   |  0.0229219  |
| gemma27b | musique     | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        400 |                0.648125 |            0.64125  |            0.006875    | -0.0195833   |  0.0350052  |
| gemma27b | musique     | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        400 |                0.648125 |            0.640625 |            0.0075      | -0.0191667   |  0.0341719  |
| gemma27b | musique     | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        449 |                0.648107 |            0.652004 |           -0.00389755  | -0.0289532   |  0.0206013  |
| gemma27b | musique     | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        449 |                0.648107 |            0.65052  |           -0.00241277  | -0.027654    |  0.0220861  |
| gemma27b | musique     | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        449 |                0.648107 |            0.64922  |           -0.00111359  | -0.0280252   |  0.0259883  |
| gemma27b | musique     | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        449 |                0.648107 |            0.63827  |            0.00983667  | -0.017075    |  0.0361915  |
| gemma27b | musique     | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        400 |                0.648125 |            0.664583 |           -0.0164583   | -0.0408333   |  0.00791667 |
| gemma27b | musique     | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        450 |                1        |            1        |            0           |  0           |  0          |
| gemma27b | musique     | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        400 |                0.648125 |            0.651042 |           -0.00291667  | -0.0291667   |  0.0239583  |
| gemma27b | musique     | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        450 |                0.650556 |            0.657963 |           -0.00740741  | -0.0305556   |  0.0157407  |
| gemma27b | musique     | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        450 |                0.649074 |            0.656481 |           -0.00740741  | -0.0314815   |  0.0159259  |

### Answer accuracy and paired flips (pre-adjudication) [table: data/answer_summary.csv, data/answer_paired_flips.csv]

Automatic semantic labels, not adjudicated accuracy (spec 3.3) -- see `needs_adjudication_flag` counts in [table: data/answer_run_level.csv] before citing.

**Per-cell summary**

| model    | dataset     | condition                                         |   n_total_rows |   n_primary_denominator |   n_technical_failures_excluded |   n_valid_answer |   pct_valid_answer |   n_semantic_correct |   pct_semantic_correct |   n_needs_adjudication |   exact_match_mean |   token_f1_mean |   gold_containment_pct_diagnostic_only |
|:---------|:------------|:--------------------------------------------------|---------------:|------------------------:|--------------------------------:|-----------------:|-------------------:|---------------------:|-----------------------:|-----------------------:|-------------------:|----------------:|---------------------------------------:|
| gemma27b | hotpot      | NaturalClean                                      |            450 |                     450 |                               0 |              449 |            99.7778 |                  169 |                37.5556 |                      4 |          0.148889  |        0.256952 |                                80      |
| gemma27b | hotpot      | NaturalNeutral                                    |            450 |                     450 |                               0 |              449 |            99.7778 |                  161 |                35.7778 |                      7 |          0.126667  |        0.236878 |                                80.2222 |
| gemma27b | hotpot      | NaturalContinueSearch                             |            450 |                     450 |                               0 |              450 |           100      |                  174 |                38.6667 |                      5 |          0.131111  |        0.249836 |                                79.3333 |
| gemma27b | hotpot      | Natural_epistemic_confirmation                    |            450 |                     450 |                               0 |              449 |            99.7778 |                  171 |                38      |                      6 |          0.131111  |        0.252124 |                                80.4444 |
| gemma27b | hotpot      | Natural_search_match_cue                          |            450 |                     450 |                               0 |              448 |            99.5556 |                  169 |                37.5556 |                      8 |          0.122222  |        0.240224 |                                79.5556 |
| gemma27b | hotpot      | Natural_source_coverage                           |            450 |                     450 |                               0 |              448 |            99.5556 |                  170 |                37.7778 |                      5 |          0.135556  |        0.251665 |                                79.7778 |
| gemma27b | hotpot      | Natural_epistemic_confirmation_source             |            450 |                     450 |                               0 |              449 |            99.7778 |                  187 |                41.5556 |                      5 |          0.144444  |        0.25737  |                                81.1111 |
| gemma27b | hotpot      | Natural_search_match_source                       |            450 |                     450 |                               0 |              447 |            99.3333 |                  180 |                40      |                      9 |          0.14      |        0.256505 |                                80.8889 |
| gemma27b | hotpot      | Natural_PreRegisteredSufficiency                  |            450 |                     450 |                               0 |              450 |           100      |                  172 |                38.2222 |                      6 |          0.131111  |        0.249453 |                                79.5556 |
| gemma27b | hotpot      | Natural_PreRegisteredSufficiencySaturation        |            450 |                     450 |                               0 |              449 |            99.7778 |                  165 |                36.6667 |                      9 |          0.14      |        0.257983 |                                78.8889 |
| gemma27b | hotpot      | OracleComplete_Neutral                            |            450 |                     450 |                               0 |              449 |            99.7778 |                  191 |                42.4444 |                      6 |          0.153333  |        0.279294 |                                88.2222 |
| gemma27b | hotpot      | OracleComplete_PreRegisteredSufficiencySaturation |            450 |                     450 |                               0 |              449 |            99.7778 |                  191 |                42.4444 |                      6 |          0.142222  |        0.279668 |                                87.1111 |
| gemma27b | 2wiki       | NaturalClean                                      |            450 |                     450 |                               0 |              448 |            99.5556 |                  156 |                34.6667 |                     13 |          0.1       |        0.248987 |                                76.6667 |
| gemma27b | 2wiki       | NaturalNeutral                                    |            450 |                     450 |                               0 |              448 |            99.5556 |                  166 |                36.8889 |                      9 |          0.102222  |        0.254776 |                                78.8889 |
| gemma27b | 2wiki       | NaturalContinueSearch                             |            450 |                     450 |                               0 |              449 |            99.7778 |                  160 |                35.5556 |                     12 |          0.102222  |        0.255607 |                                76.6667 |
| gemma27b | 2wiki       | Natural_epistemic_confirmation                    |            450 |                     450 |                               0 |              449 |            99.7778 |                  147 |                32.6667 |                      9 |          0.0933333 |        0.241627 |                                75.3333 |
| gemma27b | 2wiki       | Natural_search_match_cue                          |            450 |                     450 |                               0 |              447 |            99.3333 |                  152 |                33.7778 |                     13 |          0.104444  |        0.251413 |                                76.4444 |
| gemma27b | 2wiki       | Natural_source_coverage                           |            450 |                     450 |                               0 |              446 |            99.1111 |                  154 |                34.2222 |                     10 |          0.102222  |        0.24167  |                                75.7778 |
| gemma27b | 2wiki       | Natural_epistemic_confirmation_source             |            450 |                     450 |                               0 |              448 |            99.5556 |                  161 |                35.7778 |                      9 |          0.0888889 |        0.2415   |                                74.6667 |
| gemma27b | 2wiki       | Natural_search_match_source                       |            450 |                     450 |                               0 |              450 |           100      |                  146 |                32.4444 |                     11 |          0.104444  |        0.248359 |                                78      |
| gemma27b | 2wiki       | Natural_PreRegisteredSufficiency                  |            450 |                     450 |                               0 |              449 |            99.7778 |                  158 |                35.1111 |                     11 |          0.1       |        0.248214 |                                76      |
| gemma27b | 2wiki       | Natural_PreRegisteredSufficiencySaturation        |            350 |                     350 |                               0 |              349 |            99.7143 |                  130 |                37.1429 |                      8 |          0.108571  |        0.271404 |                                77.1429 |
| gemma27b | 2wiki       | OracleComplete_Neutral                            |            450 |                     450 |                               0 |              450 |           100      |                  181 |                40.2222 |                     13 |          0.113333  |        0.27469  |                                84.6667 |
| gemma27b | 2wiki       | OracleComplete_PreRegisteredSufficiencySaturation |            450 |                     450 |                               0 |              449 |            99.7778 |                  181 |                40.2222 |                     13 |          0.128889  |        0.28393  |                                85.1111 |
| gemma27b | musique     | NaturalClean                                      |            450 |                     450 |                               0 |              448 |            99.5556 |                   96 |                21.3333 |                      6 |          0.0333333 |        0.131018 |                                53.5556 |
| gemma27b | musique     | NaturalNeutral                                    |            400 |                     400 |                               0 |              397 |            99.25   |                   74 |                18.5    |                     10 |          0.0475    |        0.13373  |                                52.25   |
| gemma27b | musique     | NaturalContinueSearch                             |            450 |                     450 |                               0 |              447 |            99.3333 |                   87 |                19.3333 |                      7 |          0.0511111 |        0.139916 |                                54.4444 |
| gemma27b | musique     | Natural_epistemic_confirmation                    |            450 |                     450 |                               0 |              448 |            99.5556 |                   84 |                18.6667 |                      9 |          0.0444444 |        0.133262 |                                53.3333 |
| gemma27b | musique     | Natural_search_match_cue                          |            450 |                     450 |                               0 |              450 |           100      |                   84 |                18.6667 |                      7 |          0.0533333 |        0.142074 |                                54.4444 |
| gemma27b | musique     | Natural_source_coverage                           |            450 |                     450 |                               0 |              449 |            99.7778 |                   79 |                17.5556 |                     13 |          0.0488889 |        0.139911 |                                54.4444 |
| gemma27b | musique     | Natural_epistemic_confirmation_source             |            450 |                     450 |                               0 |              448 |            99.5556 |                   78 |                17.3333 |                      8 |          0.0466667 |        0.129245 |                                52.2222 |
| gemma27b | musique     | Natural_search_match_source                       |            450 |                     450 |                               0 |              447 |            99.3333 |                   90 |                20      |                      6 |          0.0488889 |        0.145854 |                                53.1111 |
| gemma27b | musique     | Natural_PreRegisteredSufficiency                  |            450 |                     450 |                               0 |              447 |            99.3333 |                   82 |                18.2222 |                     11 |          0.0555556 |        0.149401 |                                52.8889 |
| gemma27b | musique     | Natural_PreRegisteredSufficiencySaturation        |            450 |                     450 |                               0 |              448 |            99.5556 |                   89 |                19.7778 |                      9 |          0.0555556 |        0.150167 |                                53.3333 |
| gemma27b | musique     | OracleComplete_Neutral                            |            450 |                     450 |                               0 |              450 |           100      |                  106 |                23.5556 |                      6 |          0.0755556 |        0.166632 |                                70.4444 |
| gemma27b | musique     | OracleComplete_PreRegisteredSufficiencySaturation |            450 |                     450 |                               0 |              450 |           100      |                  116 |                25.7778 |                      4 |          0.0644444 |        0.185929 |                                71.1111 |
| gemma27b | multihoprag | NaturalClean                                      |            450 |                     450 |                               0 |              448 |            99.5556 |                  148 |                32.8889 |                      9 |          0.224444  |        0.247533 |                                52      |
| gemma27b | multihoprag | NaturalNeutral                                    |            450 |                     450 |                               0 |              447 |            99.3333 |                  148 |                32.8889 |                      5 |          0.195556  |        0.225326 |                                51.5556 |
| gemma27b | multihoprag | NaturalContinueSearch                             |            450 |                     450 |                               0 |              447 |            99.3333 |                  128 |                28.4444 |                      9 |          0.168889  |        0.196933 |                                50      |
| gemma27b | multihoprag | Natural_epistemic_confirmation                    |            450 |                     450 |                               0 |              450 |           100      |                  141 |                31.3333 |                      6 |          0.208889  |        0.235379 |                                53.3333 |
| gemma27b | multihoprag | Natural_search_match_cue                          |            450 |                     450 |                               0 |              448 |            99.5556 |                  129 |                28.6667 |                     11 |          0.184444  |        0.207362 |                                51.1111 |
| gemma27b | multihoprag | Natural_source_coverage                           |            450 |                     450 |                               0 |              448 |            99.5556 |                  130 |                28.8889 |                      9 |          0.188889  |        0.212539 |                                50      |
| gemma27b | multihoprag | Natural_epistemic_confirmation_source             |            450 |                     450 |                               0 |              448 |            99.5556 |                  131 |                29.1111 |                      9 |          0.186667  |        0.216826 |                                50.2222 |
| gemma27b | multihoprag | Natural_search_match_source                       |            450 |                     450 |                               0 |              449 |            99.7778 |                  135 |                30      |                     13 |          0.182222  |        0.212067 |                                51.5556 |
| gemma27b | multihoprag | Natural_PreRegisteredSufficiency                  |            450 |                     450 |                               0 |              448 |            99.5556 |                  129 |                28.6667 |                     10 |          0.202222  |        0.222615 |                                53.7778 |
| gemma27b | multihoprag | Natural_PreRegisteredSufficiencySaturation        |            450 |                     450 |                               0 |              447 |            99.3333 |                  146 |                32.4444 |                      6 |          0.197778  |        0.226899 |                                55.5556 |
| gemma27b | multihoprag | OracleComplete_Neutral                            |            450 |                     450 |                               0 |              449 |            99.7778 |                  125 |                27.7778 |                      4 |          0.162222  |        0.191757 |                                50.2222 |
| gemma27b | multihoprag | OracleComplete_PreRegisteredSufficiencySaturation |            450 |                     450 |                               0 |              447 |            99.3333 |                  138 |                30.6667 |                      9 |          0.2       |        0.226598 |                                52.4444 |

**Paired answer flips**

| model    | dataset     | condition_a                    | condition_b                                       |   n_pairs |   n_technical_excluded |   both_correct |   control_correct_attack_wrong |   control_wrong_attack_correct |   both_wrong |   net_harm |   net_harm_pct_points |
|:---------|:------------|:-------------------------------|:--------------------------------------------------|----------:|-----------------------:|---------------:|-------------------------------:|-------------------------------:|-------------:|-----------:|----------------------:|
| gemma27b | hotpot      | NaturalClean                   | NaturalNeutral                                    |       450 |                      0 |            124 |                             45 |                             37 |          244 |          8 |              1.77778  |
| gemma27b | hotpot      | NaturalNeutral                 | Natural_epistemic_confirmation                    |       450 |                      0 |            120 |                             41 |                             51 |          238 |        -10 |             -2.22222  |
| gemma27b | hotpot      | NaturalNeutral                 | Natural_search_match_cue                          |       450 |                      0 |            115 |                             46 |                             54 |          235 |         -8 |             -1.77778  |
| gemma27b | hotpot      | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       450 |                      0 |            116 |                             45 |                             56 |          233 |        -11 |             -2.44444  |
| gemma27b | hotpot      | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |            112 |                             49 |                             53 |          236 |         -4 |             -0.888889 |
| gemma27b | hotpot      | NaturalClean                   | Natural_epistemic_confirmation                    |       450 |                      0 |            128 |                             41 |                             43 |          238 |         -2 |             -0.444444 |
| gemma27b | hotpot      | NaturalClean                   | Natural_search_match_cue                          |       450 |                      0 |            130 |                             39 |                             39 |          242 |          0 |              0        |
| gemma27b | hotpot      | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       450 |                      0 |            129 |                             40 |                             43 |          238 |         -3 |             -0.666667 |
| gemma27b | hotpot      | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |            127 |                             42 |                             38 |          243 |          4 |              0.888889 |
| gemma27b | hotpot      | NaturalNeutral                 | NaturalContinueSearch                             |       450 |                      0 |            120 |                             41 |                             54 |          235 |        -13 |             -2.88889  |
| gemma27b | hotpot      | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       450 |                      0 |            153 |                             38 |                             38 |          221 |          0 |              0        |
| gemma27b | hotpot      | NaturalNeutral                 | Natural_source_coverage                           |       450 |                      0 |            118 |                             43 |                             52 |          237 |         -9 |             -2        |
| gemma27b | hotpot      | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       450 |                      0 |            135 |                             36 |                             52 |          227 |        -16 |             -3.55556  |
| gemma27b | hotpot      | Natural_search_match_cue       | Natural_search_match_source                       |       450 |                      0 |            132 |                             37 |                             48 |          233 |        -11 |             -2.44444  |
| gemma27b | 2wiki       | NaturalClean                   | NaturalNeutral                                    |       450 |                      0 |            115 |                             41 |                             51 |          243 |        -10 |             -2.22222  |
| gemma27b | 2wiki       | NaturalNeutral                 | Natural_epistemic_confirmation                    |       450 |                      0 |            111 |                             55 |                             36 |          248 |         19 |              4.22222  |
| gemma27b | 2wiki       | NaturalNeutral                 | Natural_search_match_cue                          |       450 |                      0 |            113 |                             53 |                             39 |          245 |         14 |              3.11111  |
| gemma27b | 2wiki       | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       450 |                      0 |            110 |                             56 |                             48 |          236 |          8 |              1.77778  |
| gemma27b | 2wiki       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       350 |                      0 |             93 |                             40 |                             37 |          180 |          3 |              0.857143 |
| gemma27b | 2wiki       | NaturalClean                   | Natural_epistemic_confirmation                    |       450 |                      0 |            109 |                             47 |                             38 |          256 |          9 |              2        |
| gemma27b | 2wiki       | NaturalClean                   | Natural_search_match_cue                          |       450 |                      0 |            112 |                             44 |                             40 |          254 |          4 |              0.888889 |
| gemma27b | 2wiki       | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       450 |                      0 |            116 |                             40 |                             42 |          252 |         -2 |             -0.444444 |
| gemma27b | 2wiki       | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       350 |                      0 |             96 |                             29 |                             34 |          191 |         -5 |             -1.42857  |
| gemma27b | 2wiki       | NaturalNeutral                 | NaturalContinueSearch                             |       450 |                      0 |            111 |                             55 |                             49 |          235 |          6 |              1.33333  |
| gemma27b | 2wiki       | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       450 |                      0 |            144 |                             37 |                             37 |          232 |          0 |              0        |
| gemma27b | 2wiki       | NaturalNeutral                 | Natural_source_coverage                           |       450 |                      0 |            110 |                             56 |                             44 |          240 |         12 |              2.66667  |
| gemma27b | 2wiki       | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       450 |                      0 |            107 |                             40 |                             54 |          249 |        -14 |             -3.11111  |
| gemma27b | 2wiki       | Natural_search_match_cue       | Natural_search_match_source                       |       450 |                      0 |            112 |                             40 |                             34 |          264 |          6 |              1.33333  |
| gemma27b | musique     | NaturalClean                   | NaturalNeutral                                    |       400 |                      0 |             47 |                             37 |                             27 |          289 |         10 |              2.5      |
| gemma27b | musique     | NaturalNeutral                 | Natural_epistemic_confirmation                    |       400 |                      0 |             41 |                             33 |                             33 |          293 |          0 |              0        |
| gemma27b | musique     | NaturalNeutral                 | Natural_search_match_cue                          |       400 |                      0 |             43 |                             31 |                             34 |          292 |         -3 |             -0.75     |
| gemma27b | musique     | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       400 |                      0 |             35 |                             39 |                             35 |          291 |          4 |              1        |
| gemma27b | musique     | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       400 |                      0 |             37 |                             37 |                             41 |          285 |         -4 |             -1        |
| gemma27b | musique     | NaturalClean                   | Natural_epistemic_confirmation                    |       450 |                      0 |             55 |                             41 |                             29 |          325 |         12 |              2.66667  |
| gemma27b | musique     | NaturalClean                   | Natural_search_match_cue                          |       450 |                      0 |             56 |                             40 |                             28 |          326 |         12 |              2.66667  |
| gemma27b | musique     | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       450 |                      0 |             47 |                             49 |                             35 |          319 |         14 |              3.11111  |
| gemma27b | musique     | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |             53 |                             43 |                             36 |          318 |          7 |              1.55556  |
| gemma27b | musique     | NaturalNeutral                 | NaturalContinueSearch                             |       400 |                      0 |             46 |                             28 |                             32 |          294 |         -4 |             -1        |
| gemma27b | musique     | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       450 |                      0 |             56 |                             50 |                             60 |          284 |        -10 |             -2.22222  |
| gemma27b | musique     | NaturalNeutral                 | Natural_source_coverage                           |       400 |                      0 |             42 |                             32 |                             33 |          293 |         -1 |             -0.25     |
| gemma27b | musique     | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       450 |                      0 |             47 |                             37 |                             31 |          335 |          6 |              1.33333  |
| gemma27b | musique     | Natural_search_match_cue       | Natural_search_match_source                       |       450 |                      0 |             59 |                             25 |                             31 |          335 |         -6 |             -1.33333  |
| gemma27b | multihoprag | NaturalClean                   | NaturalNeutral                                    |       450 |                      0 |             90 |                             58 |                             58 |          244 |          0 |              0        |
| gemma27b | multihoprag | NaturalNeutral                 | Natural_epistemic_confirmation                    |       450 |                      0 |             86 |                             62 |                             55 |          247 |          7 |              1.55556  |
| gemma27b | multihoprag | NaturalNeutral                 | Natural_search_match_cue                          |       450 |                      0 |             84 |                             64 |                             45 |          257 |         19 |              4.22222  |
| gemma27b | multihoprag | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       450 |                      0 |             84 |                             64 |                             45 |          257 |         19 |              4.22222  |
| gemma27b | multihoprag | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |             89 |                             59 |                             57 |          245 |          2 |              0.444444 |
| gemma27b | multihoprag | NaturalClean                   | Natural_epistemic_confirmation                    |       450 |                      0 |             85 |                             63 |                             56 |          246 |          7 |              1.55556  |
| gemma27b | multihoprag | NaturalClean                   | Natural_search_match_cue                          |       450 |                      0 |             82 |                             66 |                             47 |          255 |         19 |              4.22222  |
| gemma27b | multihoprag | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       450 |                      0 |             84 |                             64 |                             45 |          257 |         19 |              4.22222  |
| gemma27b | multihoprag | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |             86 |                             62 |                             60 |          242 |          2 |              0.444444 |
| gemma27b | multihoprag | NaturalNeutral                 | NaturalContinueSearch                             |       450 |                      0 |             83 |                             65 |                             45 |          257 |         20 |              4.44444  |
| gemma27b | multihoprag | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       450 |                      0 |             77 |                             48 |                             61 |          264 |        -13 |             -2.88889  |
| gemma27b | multihoprag | NaturalNeutral                 | Natural_source_coverage                           |       450 |                      0 |             87 |                             61 |                             43 |          259 |         18 |              4        |
| gemma27b | multihoprag | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       450 |                      0 |             74 |                             67 |                             57 |          252 |         10 |              2.22222  |
| gemma27b | multihoprag | Natural_search_match_cue       | Natural_search_match_source                       |       450 |                      0 |             75 |                             54 |                             60 |          261 |         -6 |             -1.33333  |

### Retrieval-equivalence sensitivity summary [table: data/retrieval_equivalence.csv]

Main result is the same-example end-to-end comparison (`n_readable_pairs` denominator); `n_identical_observation_sensitivity_population` is the stricter subset where retrieval was byte-identical before any cue suffix, with its own denominator (spec 5.6) -- see [table: data/retrieval_equivalence.csv] for per-pair detail.

| model    | comparison                                                                  |   n_readable_pairs |   pct_identical_doc_set |   n_identical_observation_sensitivity_population |   n_total_pairs |
|:---------|:----------------------------------------------------------------------------|-------------------:|------------------------:|-------------------------------------------------:|----------------:|
| gemma27b | NaturalClean vs NaturalNeutral                                              |               1750 |                 81.0286 |                                              765 |            1750 |
| gemma27b | NaturalClean vs Natural_PreRegisteredSufficiency                            |               1800 |                 70.6111 |                                              510 |            1800 |
| gemma27b | NaturalClean vs Natural_PreRegisteredSufficiencySaturation                  |               1700 |                 71.4706 |                                              481 |            1700 |
| gemma27b | NaturalClean vs Natural_epistemic_confirmation                              |               1800 |                 71.7222 |                                              516 |            1800 |
| gemma27b | NaturalClean vs Natural_search_match_cue                                    |               1800 |                 72.0556 |                                              546 |            1800 |
| gemma27b | NaturalNeutral vs NaturalContinueSearch                                     |               1750 |                 78.0571 |                                              692 |            1750 |
| gemma27b | NaturalNeutral vs Natural_PreRegisteredSufficiency                          |               1750 |                 71.5429 |                                              490 |            1750 |
| gemma27b | NaturalNeutral vs Natural_PreRegisteredSufficiencySaturation                |               1650 |                 72.1818 |                                              460 |            1650 |
| gemma27b | NaturalNeutral vs Natural_epistemic_confirmation                            |               1750 |                 73.4857 |                                              532 |            1750 |
| gemma27b | NaturalNeutral vs Natural_search_match_cue                                  |               1750 |                 73.9429 |                                              550 |            1750 |
| gemma27b | NaturalNeutral vs Natural_source_coverage                                   |               1750 |                 73.9429 |                                              542 |            1750 |
| gemma27b | Natural_epistemic_confirmation vs Natural_epistemic_confirmation_source     |               1800 |                 73.2778 |                                              556 |            1800 |
| gemma27b | Natural_search_match_cue vs Natural_search_match_source                     |               1800 |                 73      |                                              551 |            1800 |
| gemma27b | OracleComplete_Neutral vs OracleComplete_PreRegisteredSufficiencySaturation |               1800 |                100      |                                              793 |            1800 |

### HotpotQA -- narrative summary

Valid runs: 5400/5400 [table: data/run_inventory.csv]. Neutral PCR@1 = 30.7%, reaching 100% by step 5 [table: data/pcr_stepwise.csv].

Route mix: Gemma never calls `ResearchComplete` in any condition; every episode ends by a no-tool exit, and by PCR@5 every condition is at 100% [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv]. Cue effects can therefore only appear at steps 1-2, and the zero deltas from PCR@5 onward are a ceiling, not an absence of effect.

Fixed Sufficiency raises PCR@1 by +17.6 pp [+12.4, +22.7] and Saturation by +19.8 pp [+14.7, +24.9] over Neutral; Continue Search lowers it by -16.9 pp [-21.3, -12.2] [table: data/pcr_paired_deltas_stepwise.csv].

Saturation cuts mean search calls after d1 from 0.67 to 0.50 and supervisor re-delegation from 16.4% to 13.8% [table: data/process_metrics.csv].

Evidence loss under Saturation is +1.1 pp [-1.1, +3.3], not significant [table: data/evidence_summary.csv]. Accuracy is 37.6% (Clean) vs 36.7% (Saturation) [table: data/answer_summary.csv]; Saturation vs Neutral flips 49 correct→wrong and 53 wrong→correct (net -0.9 pp, n = 450) [table: data/answer_paired_flips.csv].

Retrieval equivalence (pooled over Gemma datasets): about 72% of Neutral/Saturation pairs saw an identical first document set [table: data/retrieval_equivalence.csv].

Supported: the sufficiency cues make Gemma stop earlier at step 1. Not supported: any loss of evidence or answer accuracy on HotpotQA. [table: data/pcr_paired_deltas_stepwise.csv]

### 2WikiMultiHopQA -- narrative summary

Valid runs: 5300/5400; two shards of this cell never ran and are reported as missing, not imputed [table: data/run_inventory.csv]. Neutral PCR@1 = 20.2%, 100% by step 5 [table: data/pcr_stepwise.csv].

Route mix: Gemma never calls `ResearchComplete` in any condition; every episode ends by a no-tool exit, and by PCR@5 every condition is at 100% [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv]. Cue effects can therefore only appear at steps 1-2, and the zero deltas from PCR@5 onward are a ceiling, not an absence of effect.

Sufficiency +20.0 pp [+15.3, +24.7] and Saturation +14.6 pp [+9.4, +19.4] at PCR@1; Continue Search -5.1 pp [-8.7, -1.6] [table: data/pcr_paired_deltas_stepwise.csv].

Search calls after d1 fall from 0.84 to 0.65 under Saturation; re-delegation falls from 16.0% to 11.7% [table: data/process_metrics.csv].

Evidence loss +1.2 pp [-0.9, +3.4], not significant [table: data/evidence_summary.csv]. Accuracy 34.7% Clean vs 37.1% Saturation [table: data/answer_summary.csv]; net flips vs Neutral +0.9 pp (40 vs 37, n = 350) [table: data/answer_paired_flips.csv].

Identical first-document sets in about 72% of Neutral/Saturation pairs, pooled over Gemma datasets [table: data/retrieval_equivalence.csv].

Supported: earlier stopping at step 1. Not supported: downstream harm. [table: data/pcr_paired_deltas_stepwise.csv]

### MuSiQue -- narrative summary

Valid runs: 5350/5400; one shard missing [table: data/run_inventory.csv]. Neutral PCR@1 = 16.2%, the lowest Gemma baseline, 100% by step 5 [table: data/pcr_stepwise.csv].

Route mix: Gemma never calls `ResearchComplete` in any condition; every episode ends by a no-tool exit, and by PCR@5 every condition is at 100% [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv]. Cue effects can therefore only appear at steps 1-2, and the zero deltas from PCR@5 onward are a ceiling, not an absence of effect.

Sufficiency +11.8 pp [+6.5, +17.0] and Saturation +17.5 pp [+12.2, +23.0] at PCR@1; Continue Search -4.5 pp [-8.8, +0.0], borderline [table: data/pcr_paired_deltas_stepwise.csv].

Search calls after d1 0.86 → 0.69; re-delegation 21.8% → 19.1% [table: data/process_metrics.csv].

Evidence loss +0.7 pp [-1.9, +3.4], not significant [table: data/evidence_summary.csv]. Accuracy 21.3% Clean vs 19.8% Saturation [table: data/answer_summary.csv]; net flips vs Neutral -1.0 pp (37 vs 41, n = 400) [table: data/answer_paired_flips.csv].

Identical first-document sets in about 72% of Neutral/Saturation pairs, pooled over Gemma datasets [table: data/retrieval_equivalence.csv].

Supported: earlier stopping. Not supported: harm; MuSiQue accuracy is low in every condition, which limits how much a cue could lower it. [table: data/pcr_paired_deltas_stepwise.csv]

### MultiHopRAG -- narrative summary

Valid runs: 5400/5400 [table: data/run_inventory.csv]. Neutral PCR@1 = 25.8%, 100% by step 5 [table: data/pcr_stepwise.csv].

Route mix: Gemma never calls `ResearchComplete` in any condition; every episode ends by a no-tool exit, and by PCR@5 every condition is at 100% [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv]. Cue effects can therefore only appear at steps 1-2, and the zero deltas from PCR@5 onward are a ceiling, not an absence of effect.

The largest Gemma effects: Sufficiency +26.2 pp [+20.2, +32.4] and Saturation +23.6 pp [+17.6, +29.6] at PCR@1; Continue Search -10.0 pp [-15.1, -5.3] [table: data/pcr_paired_deltas_stepwise.csv].

Search calls after d1 0.73 → 0.45 under Saturation; re-delegation 20.9% → 23.3% [table: data/process_metrics.csv].

Evidence loss +0.9 pp [+0.3, +1.4]: small but significant, the only significant Saturation evidence loss for Gemma [table: data/evidence_summary.csv]. Accuracy 32.9% Clean vs 32.4% Saturation [table: data/answer_summary.csv]; net flips +0.4 pp (59 vs 57, n = 450) [table: data/answer_paired_flips.csv].

Identical first-document sets in about 72% of Neutral/Saturation pairs, pooled over Gemma datasets [table: data/retrieval_equivalence.csv].

Supported: earlier stopping and a small evidence loss. Not supported: an accuracy loss. [table: data/pcr_paired_deltas_stepwise.csv]

## 7.3 GPT-OSS-20B results

### Compact PCR@k tables (all 12 conditions) [table: data/pcr_stepwise.csv]

## GPT-OSS-20B
### HotpotQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 429 | 5.4 | 64.3 | 91.6 | 100.0 |
| Neutral | 390 | 6.9 | 60.0 | 93.6 | 99.7 |
| ContinueSearch | 427 | 3.5 | 61.1 | 93.9 | 100.0 |
| Epistemic confirmation | 426 | 7.5 | 58.0 | 92.0 | 99.3 |
| Search-match cue | 404 | 10.6 | 64.9 | 92.6 | 100.0 |
| Source coverage | 422 | 10.2 | 63.3 | 96.4 | 100.0 |
| Epistemic confirmation + source | 426 | 13.6 | 66.2 | 92.7 | 99.8 |
| Search-match + source | 401 | 11.0 | 68.8 | 96.0 | 100.0 |
| PreRegisteredSufficiency | 417 | 16.1 | 71.0 | 94.7 | 100.0 |
| PreRegisteredSufficiencySaturation | 417 | 23.0 | 71.2 | 96.6 | 99.8 |
| Oracle Neutral | 426 | 8.9 | 61.7 | 94.4 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 423 | 20.1 | 75.4 | 96.2 | 99.8 |

### 2WikiMultiHopQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 448 | 6.5 | 72.3 | 95.1 | 99.8 |
| Neutral | 450 | 7.3 | 75.1 | 94.4 | 99.8 |
| ContinueSearch | 449 | 2.2 | 67.7 | 93.5 | 99.6 |
| Epistemic confirmation | 450 | 11.6 | 72.2 | 94.9 | 100.0 |
| Search-match cue | 450 | 11.1 | 73.8 | 95.8 | 99.8 |
| Source coverage | 398 | 6.8 | 73.4 | 95.5 | 99.7 |
| Epistemic confirmation + source | 448 | 13.6 | 75.7 | 94.6 | 99.6 |
| Search-match + source | 449 | 12.5 | 73.7 | 94.4 | 99.6 |
| PreRegisteredSufficiency | 450 | 21.1 | 77.6 | 97.3 | 99.6 |
| PreRegisteredSufficiencySaturation | 450 | 17.8 | 81.3 | 96.2 | 99.8 |
| Oracle Neutral | 450 | 10.4 | 75.8 | 95.1 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 450 | 23.8 | 78.4 | 96.4 | 99.8 |

### MuSiQue

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 423 | 5.4 | 61.7 | 91.7 | 99.8 |
| Neutral | 424 | 8.0 | 68.6 | 94.3 | 99.8 |
| ContinueSearch | 422 | 3.3 | 68.2 | 95.5 | 99.5 |
| Epistemic confirmation | 427 | 9.6 | 65.3 | 93.4 | 99.8 |
| Search-match cue | 425 | 8.2 | 68.5 | 95.8 | 100.0 |
| Source coverage | 423 | 11.3 | 66.0 | 92.2 | 100.0 |
| Epistemic confirmation + source | 424 | 9.7 | 65.8 | 94.1 | 100.0 |
| Search-match + source | 431 | 9.5 | 69.4 | 92.6 | 100.0 |
| PreRegisteredSufficiency | 430 | 16.7 | 71.6 | 93.5 | 100.0 |
| PreRegisteredSufficiencySaturation | 432 | 16.4 | 73.4 | 95.4 | 99.8 |
| Oracle Neutral | 426 | 8.5 | 67.1 | 93.7 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 425 | 18.6 | 77.9 | 95.5 | 100.0 |

### MultiHopRAG

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 410 | 8.3 | 54.6 | 89.5 | 100.0 |
| Neutral | 418 | 8.4 | 61.7 | 92.6 | 100.0 |
| ContinueSearch | 418 | 5.5 | 57.9 | 88.3 | 99.5 |
| Epistemic confirmation | 412 | 10.4 | 61.7 | 91.7 | 99.5 |
| Search-match cue | 419 | 12.4 | 67.5 | 91.4 | 100.0 |
| Source coverage | 418 | 9.8 | 62.2 | 92.8 | 100.0 |
| Epistemic confirmation + source | 412 | 10.2 | 63.3 | 92.2 | 100.0 |
| Search-match + source | 414 | 12.6 | 65.5 | 93.0 | 100.0 |
| PreRegisteredSufficiency | 424 | 14.6 | 70.5 | 92.7 | 100.0 |
| PreRegisteredSufficiencySaturation | 422 | 19.2 | 74.2 | 95.3 | 100.0 |
| Oracle Neutral | 426 | 9.9 | 62.7 | 92.7 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 424 | 17.0 | 75.5 | 96.0 | 100.0 |

### Paired PCR deltas at PCR@{1,5,10,30}, all comparisons [table: data/pcr_paired_deltas_stepwise.csv]

| Dataset | Comparison | n_paired | ΔPCR@1 | ΔPCR@5 | ΔPCR@10 | ΔPCR@30 |
|---|---|---|---|---|---|---|
| 2WikiMultiHopQA | Continue Search vs Natural Neutral | 449 | -5.1% | -7.6% | -0.9% | -0.2% |
| 2WikiMultiHopQA | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 448 | 2.0% | 3.6% | -0.2% | -0.4% |
| 2WikiMultiHopQA | Epistemic Confirmation vs Natural Neutral | 450 | 4.2% | -2.9% | 0.4% | 0.2% |
| 2WikiMultiHopQA | Fixed Sufficiency Saturation vs Natural Neutral | 450 | 10.4% | 6.2% | 1.8% | 0.0% |
| 2WikiMultiHopQA | Fixed Sufficiency vs Natural Neutral | 450 | 13.8% | 2.4% | 2.9% | -0.2% |
| 2WikiMultiHopQA | Natural Neutral vs Natural Clean | 448 | 0.9% | 2.9% | -0.4% | 0.0% |
| 2WikiMultiHopQA | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 448 | 14.7% | 5.4% | 2.2% | -0.2% |
| 2WikiMultiHopQA | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 448 | 11.4% | 8.9% | 1.1% | 0.0% |
| 2WikiMultiHopQA | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 448 | 5.1% | 0.0% | -0.2% | 0.2% |
| 2WikiMultiHopQA | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 448 | 4.7% | 1.6% | 0.7% | 0.0% |
| 2WikiMultiHopQA | Oracle Complete-Saturation vs Oracle Complete-Neutral | 450 | 13.3% | 2.7% | 1.3% | -0.2% |
| 2WikiMultiHopQA | Search Match + Source Coverage vs Search Match | 449 | 1.3% | -0.2% | -1.3% | -0.2% |
| 2WikiMultiHopQA | Search Match vs Natural Neutral | 450 | 3.8% | -1.3% | 1.3% | 0.0% |
| 2WikiMultiHopQA | Source Coverage vs Natural Neutral | 398 | -0.3% | -1.3% | 0.8% | 0.0% |
| HotpotQA | Continue Search vs Natural Neutral | 371 | -2.7% | 1.6% | 0.5% | 0.3% |
| HotpotQA | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 406 | 6.2% | 7.1% | 0.2% | 0.5% |
| HotpotQA | Epistemic Confirmation vs Natural Neutral | 371 | 0.5% | -2.2% | -1.6% | -0.5% |
| HotpotQA | Fixed Sufficiency Saturation vs Natural Neutral | 361 | 16.9% | 10.8% | 3.0% | 0.0% |
| HotpotQA | Fixed Sufficiency vs Natural Neutral | 364 | 8.8% | 12.9% | 1.4% | 0.3% |
| HotpotQA | Natural Neutral vs Natural Clean | 376 | 1.1% | -5.1% | 1.9% | -0.3% |
| HotpotQA | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 403 | 9.9% | 5.5% | 3.0% | 0.0% |
| HotpotQA | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 399 | 18.0% | 7.3% | 5.3% | -0.3% |
| HotpotQA | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 408 | 2.0% | -5.6% | 0.2% | -0.7% |
| HotpotQA | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 384 | 5.7% | 0.5% | 1.0% | 0.0% |
| HotpotQA | Oracle Complete-Saturation vs Oracle Complete-Neutral | 406 | 11.1% | 13.5% | 1.5% | -0.2% |
| HotpotQA | Search Match + Source Coverage vs Search Match | 375 | 1.1% | 6.1% | 3.7% | 0.0% |
| HotpotQA | Search Match vs Natural Neutral | 351 | 4.0% | 3.4% | -0.6% | 0.3% |
| HotpotQA | Source Coverage vs Natural Neutral | 367 | 3.5% | 2.7% | 2.7% | 0.3% |
| MultiHopRAG | Continue Search vs Natural Neutral | 391 | -3.3% | -4.9% | -3.3% | -0.5% |
| MultiHopRAG | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 383 | 0.0% | 1.6% | 0.5% | 0.5% |
| MultiHopRAG | Epistemic Confirmation vs Natural Neutral | 385 | 1.8% | 0.3% | -1.0% | -0.5% |
| MultiHopRAG | Fixed Sufficiency Saturation vs Natural Neutral | 392 | 10.5% | 12.5% | 2.6% | 0.0% |
| MultiHopRAG | Fixed Sufficiency vs Natural Neutral | 394 | 6.1% | 7.9% | 0.3% | 0.0% |
| MultiHopRAG | Natural Neutral vs Natural Clean | 380 | 0.5% | 8.4% | 3.4% | 0.0% |
| MultiHopRAG | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 387 | 5.9% | 15.8% | 2.6% | 0.0% |
| MultiHopRAG | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 387 | 11.4% | 20.7% | 5.2% | 0.0% |
| MultiHopRAG | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 377 | 1.3% | 8.0% | 3.4% | -0.3% |
| MultiHopRAG | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 382 | 3.9% | 12.8% | 2.4% | 0.0% |
| MultiHopRAG | Oracle Complete-Saturation vs Oracle Complete-Neutral | 403 | 6.9% | 12.4% | 3.2% | 0.0% |
| MultiHopRAG | Search Match + Source Coverage vs Search Match | 388 | 0.3% | -3.6% | 2.1% | 0.0% |
| MultiHopRAG | Search Match vs Natural Neutral | 392 | 3.1% | 6.1% | -1.5% | 0.0% |
| MultiHopRAG | Source Coverage vs Natural Neutral | 389 | 0.5% | 0.5% | 1.0% | 0.0% |
| MuSiQue | Continue Search vs Natural Neutral | 403 | -5.0% | 0.2% | 1.2% | -0.2% |
| MuSiQue | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | 405 | 0.0% | 0.5% | 1.2% | 0.2% |
| MuSiQue | Epistemic Confirmation vs Natural Neutral | 405 | 1.5% | -4.0% | -1.0% | 0.0% |
| MuSiQue | Fixed Sufficiency Saturation vs Natural Neutral | 408 | 8.3% | 3.7% | 1.5% | 0.0% |
| MuSiQue | Fixed Sufficiency vs Natural Neutral | 409 | 8.8% | 3.2% | -0.7% | 0.2% |
| MuSiQue | Natural Neutral vs Natural Clean | 404 | 2.5% | 6.4% | 2.5% | 0.0% |
| MuSiQue | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | 405 | 10.9% | 10.4% | 1.7% | 0.2% |
| MuSiQue | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | 409 | 10.5% | 10.8% | 4.2% | 0.0% |
| MuSiQue | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | 404 | 5.0% | 3.7% | 2.0% | -0.2% |
| MuSiQue | Natural_search_match_cue vs Natural Clean (total end-to-end change) | 399 | 3.0% | 8.3% | 4.5% | 0.3% |
| MuSiQue | Oracle Complete-Saturation vs Oracle Complete-Neutral | 407 | 10.6% | 10.8% | 1.7% | 0.0% |
| MuSiQue | Search Match + Source Coverage vs Search Match | 408 | 1.2% | 0.5% | -3.4% | 0.0% |
| MuSiQue | Search Match vs Natural Neutral | 401 | 0.2% | 0.5% | 1.7% | 0.2% |
| MuSiQue | Source Coverage vs Natural Neutral | 399 | 3.3% | -2.5% | -2.3% | 0.3% |

### Full stepwise PCR (every step 1-30, all 12 conditions) [table: data/pcr_stepwise.csv]

Values in %. Column codes:

| Code | Condition |
|---|---|
| Clean | Clean |
| Neutral | Neutral |
| ContSearch | ContinueSearch |
| EpistConf | Epistemic confirmation |
| SearchMatch | Search-match cue |
| SrcCov | Source coverage |
| EpistConf+Src | Epistemic confirmation + source |
| SearchMatch+Src | Search-match + source |
| PreRegSuff | PreRegisteredSufficiency |
| PreRegSuffSat | PreRegisteredSufficiencySaturation |
| OracleNeutral | Oracle Neutral |
| OraclePreRegSuffSat | Oracle PreRegisteredSufficiencySaturation |

## GPT-OSS-20B / HotpotQA
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |     5.4 |          3.5 |       6.9 |         16.1 |            23   |         7.5 |            13.6 |          10.6 |              11   |     10.2 |             8.9 |                  20.1 |
|      2 |    19.1 |         14.1 |      17.7 |         30.7 |            35.7 |        18.3 |            27.2 |          22.3 |              22.7 |     21.8 |            20.4 |                  34.8 |
|      3 |    37.3 |         30.9 |      36.2 |         45.8 |            51.1 |        33.8 |            40.6 |          40.6 |              43.1 |     39.3 |            34   |                  49.9 |
|      4 |    51.5 |         48   |      49.2 |         60.4 |            61.4 |        49.3 |            54.2 |          52.5 |              57.4 |     51.7 |            50.5 |                  62.4 |
|      5 |    64.3 |         61.1 |      60   |         71   |            71.2 |        58   |            66.2 |          64.9 |              68.8 |     63.3 |            61.7 |                  75.4 |
|      6 |    73.4 |         73.3 |      69.5 |         78.9 |            79.4 |        69.7 |            73.9 |          74.8 |              78.8 |     72.5 |            73.9 |                  82.5 |
|      7 |    80.2 |         80.8 |      76.2 |         84.2 |            85.1 |        77.9 |            78.4 |          82.2 |              84.8 |     82   |            81.2 |                  89.4 |
|      8 |    84.1 |         86.7 |      83.8 |         89   |            90.9 |        83.3 |            85.9 |          85.4 |              90.5 |     90.8 |            87.6 |                  93.9 |
|      9 |    89   |         91.6 |      90.3 |         93   |            93.8 |        88.3 |            90.4 |          89.1 |              93   |     94.5 |            90.4 |                  95.3 |
|     10 |    91.6 |         93.9 |      93.6 |         94.7 |            96.6 |        92   |            92.7 |          92.6 |              96   |     96.4 |            94.4 |                  96.2 |
|     11 |    93.7 |         97   |      94.9 |         96.4 |            97.8 |        94.1 |            95.3 |          94.8 |              97.3 |     97.4 |            95.8 |                  97.4 |
|     12 |    96   |         97.9 |      96.4 |         98.1 |            98.8 |        95.3 |            96.5 |          96.3 |              98.3 |     98.1 |            97.2 |                  98.6 |
|     13 |    96.7 |         98.8 |      97.2 |         98.6 |            99   |        96.2 |            97.7 |          96.5 |              99   |     99.3 |            98.4 |                  99.1 |
|     14 |    98.1 |         99.1 |      99.2 |         99.3 |            99   |        97.2 |            98.1 |          96.5 |              99.3 |     99.5 |            99.1 |                  99.1 |
|     15 |    98.6 |         99.3 |      99.2 |         99.3 |            99.3 |        98.1 |            98.4 |          97.5 |              99.8 |     99.5 |            99.3 |                  99.3 |
|     16 |    98.8 |         99.5 |      99.2 |         99.3 |            99.5 |        98.4 |            98.8 |          98.5 |              99.8 |     99.5 |            99.5 |                  99.5 |
|     17 |    99.5 |         99.8 |      99.5 |         99.3 |            99.5 |        98.4 |            99.1 |          99   |              99.8 |     99.8 |            99.8 |                  99.8 |
|     18 |    99.8 |         99.8 |      99.5 |         99.3 |            99.5 |        98.4 |            99.3 |          99   |              99.8 |     99.8 |           100   |                  99.8 |
|     19 |    99.8 |         99.8 |      99.7 |         99.5 |            99.5 |        98.6 |            99.5 |          99.3 |              99.8 |     99.8 |           100   |                  99.8 |
|     20 |    99.8 |         99.8 |      99.7 |        100   |            99.5 |        98.8 |            99.8 |          99.8 |              99.8 |     99.8 |           100   |                  99.8 |
|     21 |    99.8 |         99.8 |      99.7 |        100   |            99.8 |        98.8 |            99.8 |          99.8 |              99.8 |     99.8 |           100   |                  99.8 |
|     22 |    99.8 |         99.8 |      99.7 |        100   |            99.8 |        99.1 |            99.8 |          99.8 |              99.8 |     99.8 |           100   |                  99.8 |
|     23 |    99.8 |         99.8 |      99.7 |        100   |            99.8 |        99.1 |            99.8 |         100   |              99.8 |     99.8 |           100   |                  99.8 |
|     24 |   100   |        100   |      99.7 |        100   |            99.8 |        99.1 |            99.8 |         100   |              99.8 |    100   |           100   |                  99.8 |
|     25 |   100   |        100   |      99.7 |        100   |            99.8 |        99.1 |            99.8 |         100   |              99.8 |    100   |           100   |                  99.8 |
|     26 |   100   |        100   |      99.7 |        100   |            99.8 |        99.3 |            99.8 |         100   |             100   |    100   |           100   |                  99.8 |
|     27 |   100   |        100   |      99.7 |        100   |            99.8 |        99.3 |            99.8 |         100   |             100   |    100   |           100   |                  99.8 |
|     28 |   100   |        100   |      99.7 |        100   |            99.8 |        99.3 |            99.8 |         100   |             100   |    100   |           100   |                  99.8 |
|     29 |   100   |        100   |      99.7 |        100   |            99.8 |        99.3 |            99.8 |         100   |             100   |    100   |           100   |                  99.8 |
|     30 |   100   |        100   |      99.7 |        100   |            99.8 |        99.3 |            99.8 |         100   |             100   |    100   |           100   |                  99.8 |

## GPT-OSS-20B / 2WikiMultiHopQA
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |     6.5 |          2.2 |       7.3 |         21.1 |            17.8 |        11.6 |            13.6 |          11.1 |              12.5 |      6.8 |            10.4 |                  23.8 |
|      2 |    23.4 |         16.7 |      23.8 |         37.6 |            38.7 |        28.4 |            32.8 |          28.9 |              28.7 |     29.9 |            35.6 |                  46.4 |
|      3 |    46.2 |         41.6 |      49.1 |         56.2 |            60.4 |        50.2 |            51.8 |          51.1 |              51.7 |     53   |            52.7 |                  61.3 |
|      4 |    61.8 |         57.2 |      63.8 |         68.4 |            71.8 |        62.7 |            67.2 |          63.3 |              62.8 |     64.8 |            66.9 |                  70.2 |
|      5 |    72.3 |         67.7 |      75.1 |         77.6 |            81.3 |        72.2 |            75.7 |          73.8 |              73.7 |     73.4 |            75.8 |                  78.4 |
|      6 |    77.9 |         76.4 |      80.4 |         84.2 |            86   |        78   |            82.1 |          80.9 |              79.5 |     79.1 |            83.1 |                  84.7 |
|      7 |    83.9 |         82   |      85.6 |         91.1 |            88.9 |        85.1 |            87.3 |          85.3 |              84.6 |     86.4 |            86.9 |                  88.4 |
|      8 |    90   |         87.8 |      88.7 |         94.4 |            92.2 |        88.4 |            90   |          91.1 |              88.2 |     91   |            90.2 |                  92   |
|      9 |    93.3 |         91.8 |      92   |         96.4 |            94.4 |        92.2 |            92.6 |          93.8 |              91.3 |     93.5 |            92.9 |                  95.1 |
|     10 |    95.1 |         93.5 |      94.4 |         97.3 |            96.2 |        94.9 |            94.6 |          95.8 |              94.4 |     95.5 |            95.1 |                  96.4 |
|     11 |    96.4 |         95.3 |      95.8 |         97.8 |            96.9 |        96.4 |            96.2 |          96.4 |              96.4 |     97   |            96   |                  97.6 |
|     12 |    96.7 |         96.9 |      96.4 |         98   |            97.6 |        97.6 |            97.1 |          97.8 |              97.1 |     98   |            96.7 |                  97.8 |
|     13 |    97.1 |         97.1 |      96.7 |         98.7 |            98   |        97.8 |            97.8 |          98.2 |              97.6 |     98.7 |            97.3 |                  98.2 |
|     14 |    97.5 |         97.6 |      97.6 |         99.1 |            98.2 |        98.4 |            98.2 |          98.7 |              97.8 |     99.5 |            97.6 |                  98.7 |
|     15 |    98   |         98   |      97.8 |         99.1 |            98.7 |        99.3 |            98.2 |          99.1 |              98.2 |     99.5 |            97.8 |                  99.1 |
|     16 |    98.2 |         98.7 |      98.4 |         99.1 |            98.9 |        99.3 |            98.4 |          99.3 |              98.4 |     99.5 |            98.2 |                  99.3 |
|     17 |    98.7 |         98.9 |      98.4 |         99.3 |            98.9 |        99.6 |            98.4 |          99.3 |              98.4 |     99.5 |            98.7 |                  99.3 |
|     18 |    99.1 |         98.9 |      98.7 |         99.3 |            99.1 |        99.6 |            98.7 |          99.3 |              98.9 |     99.5 |            99.1 |                  99.3 |
|     19 |    99.3 |         99.1 |      98.7 |         99.3 |            99.3 |        99.6 |            99.1 |          99.3 |              98.9 |     99.5 |            99.1 |                  99.3 |
|     20 |    99.3 |         99.1 |      98.9 |         99.3 |            99.3 |        99.6 |            99.1 |          99.3 |              99.3 |     99.5 |            99.1 |                  99.8 |
|     21 |    99.3 |         99.3 |      99.3 |         99.6 |            99.3 |       100   |            99.1 |          99.3 |              99.3 |     99.5 |            99.1 |                  99.8 |
|     22 |    99.3 |         99.3 |      99.6 |         99.6 |            99.6 |       100   |            99.1 |          99.6 |              99.6 |     99.7 |            99.6 |                  99.8 |
|     23 |    99.6 |         99.6 |      99.6 |         99.6 |            99.6 |       100   |            99.3 |          99.6 |              99.6 |     99.7 |            99.8 |                  99.8 |
|     24 |    99.8 |         99.6 |      99.6 |         99.6 |            99.6 |       100   |            99.6 |          99.6 |              99.6 |     99.7 |            99.8 |                  99.8 |
|     25 |    99.8 |         99.6 |      99.6 |         99.6 |            99.6 |       100   |            99.6 |          99.6 |              99.6 |     99.7 |           100   |                  99.8 |
|     26 |    99.8 |         99.6 |      99.6 |         99.6 |            99.6 |       100   |            99.6 |          99.6 |              99.6 |     99.7 |           100   |                  99.8 |
|     27 |    99.8 |         99.6 |      99.6 |         99.6 |            99.8 |       100   |            99.6 |          99.6 |              99.6 |     99.7 |           100   |                  99.8 |
|     28 |    99.8 |         99.6 |      99.8 |         99.6 |            99.8 |       100   |            99.6 |          99.6 |              99.6 |     99.7 |           100   |                  99.8 |
|     29 |    99.8 |         99.6 |      99.8 |         99.6 |            99.8 |       100   |            99.6 |          99.8 |              99.6 |     99.7 |           100   |                  99.8 |
|     30 |    99.8 |         99.6 |      99.8 |         99.6 |            99.8 |       100   |            99.6 |          99.8 |              99.6 |     99.7 |           100   |                  99.8 |

## GPT-OSS-20B / MuSiQue
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |     5.4 |          3.3 |       8   |         16.7 |            16.4 |         9.6 |             9.7 |           8.2 |               9.5 |     11.3 |             8.5 |                  18.6 |
|      2 |    18   |         15.4 |      21   |         29.5 |            27.8 |        22.2 |            21.9 |          23.5 |              22.3 |     23.4 |            21.8 |                  33.4 |
|      3 |    33.1 |         36.7 |      40.1 |         47.9 |            46.5 |        38.9 |            40.3 |          37.2 |              42.7 |     40.7 |            38.3 |                  49.4 |
|      4 |    50.1 |         54.3 |      56.4 |         61.9 |            58.3 |        55   |            53.5 |          55.5 |              54.5 |     55.3 |            54.7 |                  65.9 |
|      5 |    61.7 |         68.2 |      68.6 |         71.6 |            73.4 |        65.3 |            65.8 |          68.5 |              69.4 |     66   |            67.1 |                  77.9 |
|      6 |    71.4 |         76.8 |      75.7 |         79.1 |            81   |        75.9 |            73.8 |          77.2 |              76.1 |     76.1 |            73.9 |                  82.8 |
|      7 |    77.5 |         82   |      82.5 |         85.3 |            85.9 |        82.9 |            80.4 |          85.4 |              80.7 |     82   |            79.8 |                  87.1 |
|      8 |    85.3 |         88.4 |      87.5 |         87.7 |            90.3 |        87.8 |            87.3 |          91.1 |              85.6 |     83.9 |            86.2 |                  90.1 |
|      9 |    88.4 |         92.9 |      91.7 |         90.5 |            93.5 |        92.3 |            91.7 |          93.9 |              90.5 |     87.7 |            91.3 |                  94.1 |
|     10 |    91.7 |         95.5 |      94.3 |         93.5 |            95.4 |        93.4 |            94.1 |          95.8 |              92.6 |     92.2 |            93.7 |                  95.5 |
|     11 |    93.6 |         96.4 |      96.9 |         94.7 |            97.2 |        95.8 |            96.2 |          96.5 |              94.7 |     94.6 |            95.1 |                  96.9 |
|     12 |    95.3 |         97.2 |      98.1 |         95.6 |            97.7 |        96.7 |            97.2 |          98.1 |              95.6 |     96.2 |            95.5 |                  97.9 |
|     13 |    96.2 |         97.4 |      98.3 |         96.7 |            98.4 |        97.7 |            97.4 |          98.4 |              96.8 |     97.2 |            98.1 |                  98.6 |
|     14 |    98.1 |         98.1 |      98.6 |         98.1 |            98.6 |        98.1 |            98.1 |          99.3 |              97.9 |     98.6 |            98.6 |                  98.8 |
|     15 |    98.3 |         98.1 |      98.6 |         98.4 |            99.3 |        98.6 |            98.8 |          99.5 |              98.6 |     98.8 |            98.8 |                  99.1 |
|     16 |    98.6 |         98.3 |      98.6 |         98.4 |            99.3 |        98.8 |            99.1 |          99.8 |              99.1 |     98.8 |            99.3 |                  99.5 |
|     17 |    98.8 |         98.3 |      98.8 |         99.1 |            99.5 |        99.3 |            99.1 |          99.8 |              99.3 |     99.3 |            99.5 |                 100   |
|     18 |    98.8 |         98.6 |      98.8 |         99.1 |            99.8 |        99.3 |            99.3 |          99.8 |              99.5 |     99.3 |            99.8 |                 100   |
|     19 |    98.8 |         98.6 |      98.8 |         99.1 |            99.8 |        99.3 |            99.8 |          99.8 |              99.8 |     99.3 |            99.8 |                 100   |
|     20 |    98.8 |         98.8 |      99.1 |         99.1 |            99.8 |        99.5 |            99.8 |          99.8 |              99.8 |     99.3 |           100   |                 100   |
|     21 |    99.5 |         99.1 |      99.3 |         99.3 |            99.8 |        99.5 |            99.8 |          99.8 |              99.8 |     99.5 |           100   |                 100   |
|     22 |    99.8 |         99.1 |      99.3 |         99.5 |            99.8 |        99.5 |           100   |          99.8 |             100   |     99.8 |           100   |                 100   |
|     23 |    99.8 |         99.3 |      99.3 |         99.5 |            99.8 |        99.8 |           100   |          99.8 |             100   |     99.8 |           100   |                 100   |
|     24 |    99.8 |         99.3 |      99.5 |         99.8 |            99.8 |        99.8 |           100   |          99.8 |             100   |     99.8 |           100   |                 100   |
|     25 |    99.8 |         99.5 |      99.5 |         99.8 |            99.8 |        99.8 |           100   |         100   |             100   |    100   |           100   |                 100   |
|     26 |    99.8 |         99.5 |      99.5 |         99.8 |            99.8 |        99.8 |           100   |         100   |             100   |    100   |           100   |                 100   |
|     27 |    99.8 |         99.5 |      99.8 |         99.8 |            99.8 |        99.8 |           100   |         100   |             100   |    100   |           100   |                 100   |
|     28 |    99.8 |         99.5 |      99.8 |        100   |            99.8 |        99.8 |           100   |         100   |             100   |    100   |           100   |                 100   |
|     29 |    99.8 |         99.5 |      99.8 |        100   |            99.8 |        99.8 |           100   |         100   |             100   |    100   |           100   |                 100   |
|     30 |    99.8 |         99.5 |      99.8 |        100   |            99.8 |        99.8 |           100   |         100   |             100   |    100   |           100   |                 100   |

## GPT-OSS-20B / MultiHopRAG
|   step |   Clean |   ContSearch |   Neutral |   PreRegSuff |   PreRegSuffSat |   EpistConf |   EpistConf+Src |   SearchMatch |   SearchMatch+Src |   SrcCov |   OracleNeutral |   OraclePreRegSuffSat |
|-------:|--------:|-------------:|----------:|-------------:|----------------:|------------:|----------------:|--------------:|------------------:|---------:|----------------:|----------------------:|
|      1 |     8.3 |          5.5 |       8.4 |         14.6 |            19.2 |        10.4 |            10.2 |          12.4 |              12.6 |      9.8 |             9.9 |                  17   |
|      2 |    19   |         16.7 |      21.1 |         32.5 |            36   |        24.3 |            22.3 |          26.5 |              28.7 |     24.6 |            21.6 |                  34   |
|      3 |    29.8 |         29.9 |      35.6 |         44.1 |            49.3 |        35.9 |            36.7 |          39.6 |              39.4 |     37.3 |            37.3 |                  49.8 |
|      4 |    43.7 |         43.1 |      50.2 |         59.7 |            62.8 |        48.5 |            49.8 |          53.9 |              52.7 |     49.5 |            49.1 |                  62.7 |
|      5 |    54.6 |         57.9 |      61.7 |         70.5 |            74.2 |        61.7 |            63.3 |          67.5 |              65.5 |     62.2 |            62.7 |                  75.5 |
|      6 |    67.1 |         70.3 |      72.5 |         79   |            81.8 |        70.6 |            72.8 |          73.7 |              75.6 |     71.1 |            74.6 |                  83   |
|      7 |    76.1 |         77.3 |      78   |         85.8 |            87.2 |        78.4 |            80.1 |          80.9 |              83.1 |     81.1 |            81.9 |                  88.7 |
|      8 |    83.4 |         82.3 |      85.6 |         89.2 |            91.9 |        83.5 |            85.2 |          84.7 |              87   |     84.7 |            87.3 |                  91.3 |
|      9 |    87.1 |         86.1 |      90   |         91.5 |            94.3 |        88.8 |            90   |          87.8 |              90.8 |     89.5 |            90.4 |                  94.1 |
|     10 |    89.5 |         88.3 |      92.6 |         92.7 |            95.3 |        91.7 |            92.2 |          91.4 |              93   |     92.8 |            92.7 |                  96   |
|     11 |    91   |         90.2 |      93.8 |         94.6 |            96.4 |        93.2 |            94.4 |          93.6 |              94.9 |     93.8 |            93.4 |                  96.7 |
|     12 |    92.7 |         93.5 |      95.9 |         95.5 |            97.6 |        94.7 |            95.9 |          95   |              96.1 |     94.5 |            94.4 |                  97.6 |
|     13 |    93.9 |         94.3 |      97.1 |         96.7 |            97.6 |        95.6 |            96.1 |          95.9 |              97.1 |     96.4 |            94.6 |                  98.3 |
|     14 |    95.1 |         95.5 |      97.4 |         97.4 |            98.3 |        96.4 |            96.4 |          97.1 |              97.8 |     96.9 |            96.5 |                  98.8 |
|     15 |    96.1 |         96.2 |      99   |         97.9 |            98.6 |        97.1 |            97.8 |          97.9 |              98.1 |     97.1 |            96.9 |                  98.8 |
|     16 |    96.8 |         97.4 |      99   |         98.8 |            99.1 |        97.3 |            97.8 |          98.6 |              98.6 |     98.1 |            97.2 |                  99.1 |
|     17 |    97.1 |         97.6 |      99   |         99.1 |            99.5 |        98.1 |            98.1 |          98.8 |              98.8 |     99   |            97.9 |                  99.8 |
|     18 |    97.3 |         98.1 |      99.3 |         99.1 |            99.5 |        98.5 |            98.8 |          98.8 |              99   |     99   |            97.9 |                  99.8 |
|     19 |    98   |         98.3 |      99.3 |         99.3 |            99.8 |        98.5 |            99   |          99   |              99   |     99.3 |            98.4 |                 100   |
|     20 |    98.5 |         98.8 |      99.5 |        100   |            99.8 |        98.8 |            99   |          99.3 |              99.5 |     99.3 |            98.6 |                 100   |
|     21 |    99.3 |         98.8 |      99.5 |        100   |            99.8 |        99   |            99.3 |          99.3 |              99.5 |     99.3 |            98.6 |                 100   |
|     22 |    99.5 |         98.8 |      99.8 |        100   |            99.8 |        99   |           100   |          99.5 |              99.5 |     99.8 |            98.8 |                 100   |
|     23 |    99.5 |         98.8 |      99.8 |        100   |            99.8 |        99   |           100   |          99.5 |              99.8 |     99.8 |            98.8 |                 100   |
|     24 |    99.5 |         99.3 |     100   |        100   |            99.8 |        99   |           100   |          99.5 |              99.8 |     99.8 |            99.1 |                 100   |
|     25 |    99.5 |         99.3 |     100   |        100   |            99.8 |        99   |           100   |          99.5 |              99.8 |    100   |            99.1 |                 100   |
|     26 |    99.8 |         99.3 |     100   |        100   |            99.8 |        99.3 |           100   |         100   |              99.8 |    100   |            99.3 |                 100   |
|     27 |    99.8 |         99.5 |     100   |        100   |            99.8 |        99.3 |           100   |         100   |              99.8 |    100   |            99.5 |                 100   |
|     28 |    99.8 |         99.5 |     100   |        100   |           100   |        99.3 |           100   |         100   |             100   |    100   |            99.8 |                 100   |
|     29 |   100   |         99.5 |     100   |        100   |           100   |        99.5 |           100   |         100   |             100   |    100   |           100   |                 100   |
|     30 |   100   |         99.5 |     100   |        100   |           100   |        99.5 |           100   |         100   |             100   |    100   |           100   |                 100   |

### Route decomposition -- final terminal-route tallies, all 7 canonical routes [table: data/route_decomposition_stepwise.csv]

| model       | dataset         | condition                                 |   final_n_RESEARCH_COMPLETE |   final_n_NO_TOOL_DIRECT_EXIT |   final_n_MAX_BUDGET_EXIT |   final_n_TOOL_OR_PARSER_FAILURE |   final_n_TRACE_LOGGING_FAILURE |   final_n_EXECUTION_FAILURE |   final_n_OTHER_VALID_ROUTE |
|:------------|:----------------|:------------------------------------------|----------------------------:|------------------------------:|--------------------------:|---------------------------------:|--------------------------------:|----------------------------:|----------------------------:|
| GPT-OSS-20B | 2WikiMultiHopQA | Clean                                     |                         112 |                           335 |                         1 |                                0 |                               2 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | ContinueSearch                            |                         137 |                           310 |                         2 |                                0 |                               1 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Neutral                                   |                         108 |                           341 |                         1 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | PreRegisteredSufficiency                  |                         120 |                           328 |                         2 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | PreRegisteredSufficiencySaturation        |                         101 |                           348 |                         1 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Epistemic confirmation                    |                         119 |                           331 |                         0 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Epistemic confirmation + source           |                         108 |                           338 |                         2 |                                0 |                               2 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Search-match cue                          |                         114 |                           335 |                         1 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Search-match + source                     |                          99 |                           348 |                         2 |                                0 |                               1 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Source coverage                           |                         100 |                           297 |                         1 |                                0 |                               2 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Oracle Neutral                            |                          78 |                           372 |                         0 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | 2WikiMultiHopQA | Oracle PreRegisteredSufficiencySaturation |                          71 |                           378 |                         1 |                                0 |                               0 |                           0 |                           0 |
| GPT-OSS-20B | HotpotQA        | Clean                                     |                         115 |                           314 |                         0 |                                0 |                               3 |                          18 |                           0 |
| GPT-OSS-20B | HotpotQA        | ContinueSearch                            |                         114 |                           313 |                         0 |                                0 |                               1 |                          22 |                           0 |
| GPT-OSS-20B | HotpotQA        | Neutral                                   |                         106 |                           283 |                         1 |                                0 |                               0 |                          21 |                           0 |
| GPT-OSS-20B | HotpotQA        | PreRegisteredSufficiency                  |                          98 |                           319 |                         0 |                                0 |                               0 |                          33 |                           0 |
| GPT-OSS-20B | HotpotQA        | PreRegisteredSufficiencySaturation        |                          90 |                           326 |                         1 |                                0 |                               1 |                          32 |                           0 |
| GPT-OSS-20B | HotpotQA        | Epistemic confirmation                    |                         104 |                           319 |                         3 |                                0 |                               1 |                          23 |                           0 |
| GPT-OSS-20B | HotpotQA        | Epistemic confirmation + source           |                          92 |                           333 |                         1 |                                0 |                               0 |                          24 |                           0 |
| GPT-OSS-20B | HotpotQA        | Search-match cue                          |                         106 |                           298 |                         0 |                                0 |                               1 |                          28 |                           0 |
| GPT-OSS-20B | HotpotQA        | Search-match + source                     |                         100 |                           301 |                         0 |                                0 |                               0 |                          34 |                           0 |
| GPT-OSS-20B | HotpotQA        | Source coverage                           |                         118 |                           304 |                         0 |                                0 |                               0 |                          28 |                           0 |
| GPT-OSS-20B | HotpotQA        | Oracle Neutral                            |                          87 |                           339 |                         0 |                                0 |                               0 |                          24 |                           0 |
| GPT-OSS-20B | HotpotQA        | Oracle PreRegisteredSufficiencySaturation |                          85 |                           337 |                         1 |                                0 |                               0 |                          27 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Clean                                     |                          58 |                           352 |                         0 |                                0 |                               0 |                          40 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | ContinueSearch                            |                          49 |                           367 |                         2 |                                0 |                               0 |                          32 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Neutral                                   |                          64 |                           354 |                         0 |                                0 |                               0 |                          32 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | PreRegisteredSufficiency                  |                          77 |                           347 |                         0 |                                0 |                               0 |                          26 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | PreRegisteredSufficiencySaturation        |                          67 |                           355 |                         0 |                                0 |                               0 |                          28 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Epistemic confirmation                    |                          87 |                           323 |                         2 |                                0 |                               0 |                          35 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Epistemic confirmation + source           |                          72 |                           340 |                         0 |                                0 |                               0 |                          38 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Search-match cue                          |                          63 |                           356 |                         0 |                                0 |                               0 |                          31 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Search-match + source                     |                          52 |                           362 |                         0 |                                0 |                               0 |                          36 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Source coverage                           |                          61 |                           357 |                         0 |                                0 |                               0 |                          32 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Oracle Neutral                            |                          68 |                           358 |                         0 |                                0 |                               0 |                          24 |                           0 |
| GPT-OSS-20B | MultiHopRAG     | Oracle PreRegisteredSufficiencySaturation |                          79 |                           345 |                         0 |                                0 |                               0 |                          26 |                           0 |
| GPT-OSS-20B | MuSiQue         | Clean                                     |                          82 |                           340 |                         1 |                                0 |                               0 |                          27 |                           0 |
| GPT-OSS-20B | MuSiQue         | ContinueSearch                            |                          89 |                           331 |                         2 |                                0 |                               4 |                          24 |                           0 |
| GPT-OSS-20B | MuSiQue         | Neutral                                   |                          96 |                           327 |                         1 |                                0 |                               2 |                          24 |                           0 |
| GPT-OSS-20B | MuSiQue         | PreRegisteredSufficiency                  |                          68 |                           362 |                         0 |                                0 |                               2 |                          18 |                           0 |
| GPT-OSS-20B | MuSiQue         | PreRegisteredSufficiencySaturation        |                         102 |                           329 |                         1 |                                0 |                               1 |                          17 |                           0 |
| GPT-OSS-20B | MuSiQue         | Epistemic confirmation                    |                          86 |                           340 |                         1 |                                0 |                               1 |                          22 |                           0 |
| GPT-OSS-20B | MuSiQue         | Epistemic confirmation + source           |                          75 |                           349 |                         0 |                                0 |                               2 |                          24 |                           0 |
| GPT-OSS-20B | MuSiQue         | Search-match cue                          |                          94 |                           331 |                         0 |                                0 |                               0 |                          25 |                           0 |
| GPT-OSS-20B | MuSiQue         | Search-match + source                     |                          78 |                           353 |                         0 |                                0 |                               0 |                          19 |                           0 |
| GPT-OSS-20B | MuSiQue         | Source coverage                           |                          74 |                           349 |                         0 |                                0 |                               1 |                          26 |                           0 |
| GPT-OSS-20B | MuSiQue         | Oracle Neutral                            |                          75 |                           351 |                         0 |                                0 |                               3 |                          21 |                           0 |
| GPT-OSS-20B | MuSiQue         | Oracle PreRegisteredSufficiencySaturation |                          65 |                           360 |                         0 |                                0 |                               1 |                          24 |                           0 |

### Evidence coverage and paired evidence loss (final-report-input boundary) [table: data/evidence_summary.csv]

This is coverage of what `final_report_generation` actually received (citation-based, downstream of any redelegation/compression) -- a DIFFERENT boundary from `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage`'s local-completion-at-T_i paired evidence loss (see `tables/TABLE_INDEX.md`).

| model       | dataset     | comparison_name                                                                       | control_condition              | cue_condition                                     |   n_paired |   control_mean_coverage |   cue_mean_coverage |   paired_evidence_loss |     ci_lower |    ci_upper |
|:------------|:------------|:--------------------------------------------------------------------------------------|:-------------------------------|:--------------------------------------------------|-----------:|------------------------:|--------------------:|-----------------------:|-------------:|------------:|
| gpt_oss_20b | 2wiki       | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        448 |                0.915737 |            0.921317 |           -0.00558036  | -0.0245536   |  0.0128348  |
| gpt_oss_20b | 2wiki       | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        450 |                0.920556 |            0.931667 |           -0.0111111   | -0.0294583   |  0.00722222 |
| gpt_oss_20b | 2wiki       | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        450 |                0.920556 |            0.915556 |            0.005       | -0.015       |  0.025      |
| gpt_oss_20b | 2wiki       | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        450 |                0.920556 |            0.912222 |            0.00833333  | -0.0116667   |  0.0283333  |
| gpt_oss_20b | 2wiki       | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        450 |                0.920556 |            0.893333 |            0.0272222   |  0.00722222  |  0.0477778  |
| gpt_oss_20b | 2wiki       | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        448 |                0.915737 |            0.933594 |           -0.0178571   | -0.0368304   |  0.00111607 |
| gpt_oss_20b | 2wiki       | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        448 |                0.915737 |            0.916295 |           -0.000558036 | -0.0212054   |  0.0200893  |
| gpt_oss_20b | 2wiki       | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        448 |                0.915737 |            0.912946 |            0.00279018  | -0.0172991   |  0.0228795  |
| gpt_oss_20b | 2wiki       | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        448 |                0.915737 |            0.893973 |            0.0217634   |  0           |  0.0440848  |
| gpt_oss_20b | 2wiki       | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        449 |                0.920379 |            0.914254 |            0.00612472  | -0.0139198   |  0.0261693  |
| gpt_oss_20b | 2wiki       | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        450 |                1        |            1        |            0           |  0           |  0          |
| gpt_oss_20b | 2wiki       | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        398 |                0.919598 |            0.914573 |            0.00502513  | -0.0138348   |  0.0244975  |
| gpt_oss_20b | 2wiki       | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        448 |                0.931362 |            0.916853 |            0.0145089   | -0.00279018  |  0.031808   |
| gpt_oss_20b | 2wiki       | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        449 |                0.915367 |            0.920935 |           -0.00556793  | -0.0261693   |  0.0144766  |
| gpt_oss_20b | hotpot      | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        376 |                0.916223 |            0.914894 |            0.00132979  | -0.0199468   |  0.0239362  |
| gpt_oss_20b | hotpot      | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        371 |                0.916442 |            0.942049 |           -0.0256065   | -0.0471698   | -0.00539084 |
| gpt_oss_20b | hotpot      | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        351 |                0.91453  |            0.915954 |           -0.0014245   | -0.0242165   |  0.019943   |
| gpt_oss_20b | hotpot      | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        364 |                0.914835 |            0.914835 |            0           | -0.0247253   |  0.0247253  |
| gpt_oss_20b | hotpot      | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        361 |                0.909972 |            0.918283 |           -0.00831025  | -0.033241    |  0.0166205  |
| gpt_oss_20b | hotpot      | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        408 |                0.920343 |            0.9375   |           -0.0171569   | -0.0379902   |  0.00367647 |
| gpt_oss_20b | hotpot      | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        384 |                0.914062 |            0.910156 |            0.00390625  | -0.0182292   |  0.0260417  |
| gpt_oss_20b | hotpot      | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        403 |                0.915633 |            0.915633 |            0           | -0.0223325   |  0.0223325  |
| gpt_oss_20b | hotpot      | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        399 |                0.914787 |            0.923559 |           -0.00877193  | -0.0300752   |  0.0112782  |
| gpt_oss_20b | hotpot      | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        371 |                0.916442 |            0.916442 |            0           | -0.0229111   |  0.0229111  |
| gpt_oss_20b | hotpot      | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        406 |                1        |            1        |            0           |  0           |  0          |
| gpt_oss_20b | hotpot      | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        367 |                0.912807 |            0.910082 |            0.0027248   | -0.020436    |  0.0245232  |
| gpt_oss_20b | hotpot      | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        406 |                0.937192 |            0.916256 |            0.020936    |  0.00246305  |  0.0394397  |
| gpt_oss_20b | hotpot      | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        375 |                0.912    |            0.901333 |            0.0106667   | -0.012       |  0.0346667  |
| gpt_oss_20b | multihoprag | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        380 |                0.982237 |            0.984211 |           -0.00197368  | -0.00723684  |  0.00328947 |
| gpt_oss_20b | multihoprag | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        385 |                0.985714 |            0.985714 |            0           | -0.00519481  |  0.00519481 |
| gpt_oss_20b | multihoprag | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        392 |                0.985332 |            0.984694 |            0.000637755 | -0.00510204  |  0.00637755 |
| gpt_oss_20b | multihoprag | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        394 |                0.986041 |            0.98033  |            0.00571066  |  0.000634518 |  0.0107868  |
| gpt_oss_20b | multihoprag | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        392 |                0.984694 |            0.984056 |            0.000637755 | -0.00446429  |  0.0057398  |
| gpt_oss_20b | multihoprag | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        377 |                0.983422 |            0.985411 |           -0.00198939  | -0.00729443  |  0.00331565 |
| gpt_oss_20b | multihoprag | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        382 |                0.982984 |            0.984293 |           -0.0013089   | -0.00719895  |  0.00458115 |
| gpt_oss_20b | multihoprag | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        387 |                0.98385  |            0.98062  |            0.00322997  | -0.00258398  |  0.00904393 |
| gpt_oss_20b | multihoprag | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        387 |                0.98385  |            0.98385  |            0           | -0.00516796  |  0.00581395 |
| gpt_oss_20b | multihoprag | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        391 |                0.984655 |            0.983376 |            0.00127877  | -0.00383632  |  0.00639386 |
| gpt_oss_20b | multihoprag | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        403 |                1        |            1        |            0           |  0           |  0          |
| gpt_oss_20b | multihoprag | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        389 |                0.985219 |            0.98329  |            0.00192802  | -0.00385604  |  0.00771208 |
| gpt_oss_20b | multihoprag | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        383 |                0.98564  |            0.983681 |            0.00195822  | -0.00261097  |  0.00652742 |
| gpt_oss_20b | multihoprag | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        388 |                0.984536 |            0.981959 |            0.00257732  | -0.00193299  |  0.00708763 |
| gpt_oss_20b | musique     | Natural Neutral vs Natural Clean                                                      | NaturalClean                   | NaturalNeutral                                    |        404 |                0.75495  |            0.756394 |           -0.00144389  | -0.0290842   |  0.0266089  |
| gpt_oss_20b | musique     | Epistemic Confirmation vs Natural Neutral                                             | NaturalNeutral                 | Natural_epistemic_confirmation                    |        405 |                0.75679  |            0.761111 |           -0.00432099  | -0.03107     |  0.0220165  |
| gpt_oss_20b | musique     | Search Match vs Natural Neutral                                                       | NaturalNeutral                 | Natural_search_match_cue                          |        401 |                0.746467 |            0.758313 |           -0.0118454   | -0.0403159   |  0.0162095  |
| gpt_oss_20b | musique     | Fixed Sufficiency vs Natural Neutral                                                  | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |        409 |                0.756112 |            0.728403 |            0.0277099   | -0.00183374  |  0.057661   |
| gpt_oss_20b | musique     | Fixed Sufficiency Saturation vs Natural Neutral                                       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |        408 |                0.753472 |            0.752859 |            0.000612745 | -0.0298254   |  0.0308466  |
| gpt_oss_20b | musique     | Natural_epistemic_confirmation vs Natural Clean (total end-to-end change)             | NaturalClean                   | Natural_epistemic_confirmation                    |        404 |                0.756394 |            0.761757 |           -0.00536304  | -0.0346535   |  0.0231023  |
| gpt_oss_20b | musique     | Natural_search_match_cue vs Natural Clean (total end-to-end change)                   | NaturalClean                   | Natural_search_match_cue                          |        399 |                0.756057 |            0.755221 |            0.000835422 | -0.0290309   |  0.0315372  |
| gpt_oss_20b | musique     | Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change)           | NaturalClean                   | Natural_PreRegisteredSufficiency                  |        405 |                0.755761 |            0.734568 |            0.0211934   | -0.0063786   |  0.0491821  |
| gpt_oss_20b | musique     | Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |        409 |                0.757131 |            0.758557 |           -0.00142624  | -0.0295436   |  0.0277099  |
| gpt_oss_20b | musique     | Continue Search vs Natural Neutral                                                    | NaturalNeutral                 | NaturalContinueSearch                             |        403 |                0.751654 |            0.773573 |           -0.0219189   | -0.049421    |  0.00537634 |
| gpt_oss_20b | musique     | Oracle Complete-Saturation vs Oracle Complete-Neutral                                 | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |        407 |                1        |            1        |            0           |  0           |  0          |
| gpt_oss_20b | musique     | Source Coverage vs Natural Neutral                                                    | NaturalNeutral                 | Natural_source_coverage                           |        399 |                0.751044 |            0.751462 |           -0.000417711 | -0.0263158   |  0.0261069  |
| gpt_oss_20b | musique     | Epistemic Confirmation + Source Coverage vs Epistemic Confirmation                    | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |        405 |                0.767901 |            0.77284  |           -0.00493827  | -0.0325103   |  0.0226337  |
| gpt_oss_20b | musique     | Search Match + Source Coverage vs Search Match                                        | Natural_search_match_cue       | Natural_search_match_source                       |        408 |                0.757149 |            0.756536 |            0.000612745 | -0.0265523   |  0.0281863  |

### Answer accuracy and paired flips (pre-adjudication) [table: data/answer_summary.csv, data/answer_paired_flips.csv]

Automatic semantic labels, not adjudicated accuracy (spec 3.3) -- see `needs_adjudication_flag` counts in [table: data/answer_run_level.csv] before citing.

**Per-cell summary**

| model       | dataset     | condition                                         |   n_total_rows |   n_primary_denominator |   n_technical_failures_excluded |   n_valid_answer |   pct_valid_answer |   n_semantic_correct |   pct_semantic_correct |   n_needs_adjudication |   exact_match_mean |   token_f1_mean |   gold_containment_pct_diagnostic_only |
|:------------|:------------|:--------------------------------------------------|---------------:|------------------------:|--------------------------------:|-----------------:|-------------------:|---------------------:|-----------------------:|-----------------------:|-------------------:|----------------:|---------------------------------------:|
| gpt_oss_20b | hotpot      | NaturalClean                                      |            432 |                     432 |                               0 |              427 |            98.8426 |                  165 |                38.1944 |                      9 |          0.134259  |        0.261909 |                                71.0648 |
| gpt_oss_20b | hotpot      | NaturalNeutral                                    |            390 |                     390 |                               0 |              388 |            99.4872 |                  148 |                37.9487 |                      7 |          0.133333  |        0.249079 |                                73.5897 |
| gpt_oss_20b | hotpot      | NaturalContinueSearch                             |            428 |                     428 |                               0 |              424 |            99.0654 |                  155 |                36.215  |                      9 |          0.147196  |        0.268203 |                                75.7009 |
| gpt_oss_20b | hotpot      | Natural_epistemic_confirmation                    |            427 |                     427 |                               0 |              424 |            99.2974 |                  175 |                40.9836 |                      6 |          0.147541  |        0.286526 |                                75.1756 |
| gpt_oss_20b | hotpot      | Natural_search_match_cue                          |            405 |                     405 |                               0 |              404 |            99.7531 |                  150 |                37.037  |                      3 |          0.123457  |        0.244691 |                                74.321  |
| gpt_oss_20b | hotpot      | Natural_source_coverage                           |            422 |                     422 |                               0 |              420 |            99.5261 |                  167 |                39.5735 |                      8 |          0.135071  |        0.270617 |                                75.3555 |
| gpt_oss_20b | hotpot      | Natural_epistemic_confirmation_source             |            426 |                     426 |                               0 |              425 |            99.7653 |                  176 |                41.3146 |                      4 |          0.157277  |        0.283379 |                                74.6479 |
| gpt_oss_20b | hotpot      | Natural_search_match_source                       |            401 |                     401 |                               0 |              399 |            99.5012 |                  150 |                37.4065 |                      4 |          0.127182  |        0.247443 |                                73.5661 |
| gpt_oss_20b | hotpot      | Natural_PreRegisteredSufficiency                  |            417 |                     417 |                               0 |              413 |            99.0408 |                  152 |                36.4508 |                      8 |          0.143885  |        0.259055 |                                70.5036 |
| gpt_oss_20b | hotpot      | Natural_PreRegisteredSufficiencySaturation        |            418 |                     418 |                               0 |              418 |           100      |                  171 |                40.9091 |                      4 |          0.133971  |        0.263505 |                                76.3158 |
| gpt_oss_20b | hotpot      | OracleComplete_Neutral                            |            426 |                     426 |                               0 |              425 |            99.7653 |                  173 |                40.6103 |                      7 |          0.140845  |        0.28441  |                                76.2911 |
| gpt_oss_20b | hotpot      | OracleComplete_PreRegisteredSufficiencySaturation |            423 |                     423 |                               0 |              421 |            99.5272 |                  182 |                43.026  |                      6 |          0.148936  |        0.294645 |                                78.2506 |
| gpt_oss_20b | 2wiki       | NaturalClean                                      |            450 |                     450 |                               0 |              448 |            99.5556 |                  161 |                35.7778 |                      4 |          0.0844444 |        0.2315   |                                74.2222 |
| gpt_oss_20b | 2wiki       | NaturalNeutral                                    |            450 |                     450 |                               0 |              448 |            99.5556 |                  153 |                34      |                      3 |          0.0844444 |        0.224085 |                                74.4444 |
| gpt_oss_20b | 2wiki       | NaturalContinueSearch                             |            450 |                     450 |                               0 |              448 |            99.5556 |                  154 |                34.2222 |                      6 |          0.08      |        0.218206 |                                74.8889 |
| gpt_oss_20b | 2wiki       | Natural_epistemic_confirmation                    |            450 |                     450 |                               0 |              447 |            99.3333 |                  161 |                35.7778 |                      4 |          0.0866667 |        0.225826 |                                75.7778 |
| gpt_oss_20b | 2wiki       | Natural_search_match_cue                          |            450 |                     450 |                               0 |              448 |            99.5556 |                  156 |                34.6667 |                      5 |          0.0866667 |        0.228429 |                                74.8889 |
| gpt_oss_20b | 2wiki       | Natural_source_coverage                           |            400 |                     400 |                               0 |              396 |            99      |                  151 |                37.75   |                      5 |          0.0875    |        0.239178 |                                73.75   |
| gpt_oss_20b | 2wiki       | Natural_epistemic_confirmation_source             |            450 |                     450 |                               0 |              449 |            99.7778 |                  156 |                34.6667 |                      4 |          0.0866667 |        0.232939 |                                74.8889 |
| gpt_oss_20b | 2wiki       | Natural_search_match_source                       |            450 |                     450 |                               0 |              444 |            98.6667 |                  143 |                31.7778 |                      7 |          0.0688889 |        0.20663  |                                73.7778 |
| gpt_oss_20b | 2wiki       | Natural_PreRegisteredSufficiency                  |            450 |                     450 |                               0 |              446 |            99.1111 |                  165 |                36.6667 |                      5 |          0.0933333 |        0.239476 |                                75.5556 |
| gpt_oss_20b | 2wiki       | Natural_PreRegisteredSufficiencySaturation        |            450 |                     450 |                               0 |              448 |            99.5556 |                  165 |                36.6667 |                      5 |          0.0844444 |        0.233004 |                                73.5556 |
| gpt_oss_20b | 2wiki       | OracleComplete_Neutral                            |            450 |                     450 |                               0 |              447 |            99.3333 |                  173 |                38.4444 |                      5 |          0.0955556 |        0.24841  |                                78.6667 |
| gpt_oss_20b | 2wiki       | OracleComplete_PreRegisteredSufficiencySaturation |            450 |                     450 |                               0 |              449 |            99.7778 |                  179 |                39.7778 |                      1 |          0.0888889 |        0.245389 |                                78.2222 |
| gpt_oss_20b | musique     | NaturalClean                                      |            423 |                     423 |                               0 |              422 |            99.7636 |                   75 |                17.7305 |                      2 |          0.0425532 |        0.134971 |                                39.0071 |
| gpt_oss_20b | musique     | NaturalNeutral                                    |            426 |                     426 |                               0 |              420 |            98.5915 |                   72 |                16.9014 |                      6 |          0.028169  |        0.118441 |                                39.4366 |
| gpt_oss_20b | musique     | NaturalContinueSearch                             |            426 |                     426 |                               0 |              424 |            99.5305 |                   65 |                15.2582 |                      3 |          0.0375587 |        0.124574 |                                40.1408 |
| gpt_oss_20b | musique     | Natural_epistemic_confirmation                    |            428 |                     428 |                               0 |              427 |            99.7664 |                   79 |                18.4579 |                      3 |          0.0303738 |        0.133028 |                                42.9907 |
| gpt_oss_20b | musique     | Natural_search_match_cue                          |            425 |                     425 |                               0 |              419 |            98.5882 |                   67 |                15.7647 |                      8 |          0.0235294 |        0.109363 |                                39.0588 |
| gpt_oss_20b | musique     | Natural_source_coverage                           |            424 |                     424 |                               0 |              422 |            99.5283 |                   56 |                13.2075 |                      2 |          0.0235849 |        0.10355  |                                34.9057 |
| gpt_oss_20b | musique     | Natural_epistemic_confirmation_source             |            426 |                     426 |                               0 |              425 |            99.7653 |                   61 |                14.3192 |                      4 |          0.028169  |        0.113345 |                                39.2019 |
| gpt_oss_20b | musique     | Natural_search_match_source                       |            431 |                     431 |                               0 |              430 |            99.768  |                   69 |                16.0093 |                      5 |          0.0324826 |        0.133405 |                                38.051  |
| gpt_oss_20b | musique     | Natural_PreRegisteredSufficiency                  |            432 |                     432 |                               0 |              430 |            99.537  |                   60 |                13.8889 |                      3 |          0.0277778 |        0.120137 |                                37.963  |
| gpt_oss_20b | musique     | Natural_PreRegisteredSufficiencySaturation        |            433 |                     433 |                               0 |              431 |            99.5381 |                   61 |                14.0878 |                      7 |          0.0254042 |        0.116158 |                                40.6467 |
| gpt_oss_20b | musique     | OracleComplete_Neutral                            |            429 |                     429 |                               0 |              428 |            99.7669 |                   69 |                16.0839 |                      2 |          0.037296  |        0.117478 |                                41.0256 |
| gpt_oss_20b | musique     | OracleComplete_PreRegisteredSufficiencySaturation |            426 |                     426 |                               0 |              423 |            99.2958 |                   71 |                16.6667 |                      2 |          0.0352113 |        0.12768  |                                43.1925 |
| gpt_oss_20b | multihoprag | NaturalClean                                      |            410 |                     410 |                               0 |              407 |            99.2683 |                  148 |                36.0976 |                      4 |          0.287805  |        0.339913 |                                51.9512 |
| gpt_oss_20b | multihoprag | NaturalNeutral                                    |            418 |                     418 |                               0 |              411 |            98.3254 |                  151 |                36.1244 |                      6 |          0.282297  |        0.332049 |                                49.5215 |
| gpt_oss_20b | multihoprag | NaturalContinueSearch                             |            418 |                     418 |                               0 |              414 |            99.0431 |                  148 |                35.4067 |                      4 |          0.27512   |        0.327408 |                                49.0431 |
| gpt_oss_20b | multihoprag | Natural_epistemic_confirmation                    |            412 |                     412 |                               0 |              408 |            99.0291 |                  154 |                37.3786 |                      4 |          0.288835  |        0.337815 |                                52.6699 |
| gpt_oss_20b | multihoprag | Natural_search_match_cue                          |            419 |                     419 |                               0 |              417 |            99.5227 |                  155 |                36.9928 |                      2 |          0.303103  |        0.350153 |                                50.8353 |
| gpt_oss_20b | multihoprag | Natural_source_coverage                           |            418 |                     418 |                               0 |              410 |            98.0861 |                  158 |                37.799  |                      9 |          0.301435  |        0.346525 |                                51.1962 |
| gpt_oss_20b | multihoprag | Natural_epistemic_confirmation_source             |            412 |                     412 |                               0 |              409 |            99.2718 |                  162 |                39.3204 |                      4 |          0.320388  |        0.372401 |                                55.5825 |
| gpt_oss_20b | multihoprag | Natural_search_match_source                       |            414 |                     414 |                               0 |              412 |            99.5169 |                  157 |                37.9227 |                      3 |          0.289855  |        0.344012 |                                52.8986 |
| gpt_oss_20b | multihoprag | Natural_PreRegisteredSufficiency                  |            424 |                     424 |                               0 |              416 |            98.1132 |                  156 |                36.7925 |                     10 |          0.299528  |        0.347972 |                                53.7736 |
| gpt_oss_20b | multihoprag | Natural_PreRegisteredSufficiencySaturation        |            422 |                     422 |                               0 |              417 |            98.8152 |                  152 |                36.019  |                      5 |          0.28673   |        0.333918 |                                50.4739 |
| gpt_oss_20b | multihoprag | OracleComplete_Neutral                            |            426 |                     426 |                               0 |              419 |            98.3568 |                  159 |                37.3239 |                     10 |          0.29108   |        0.338041 |                                51.4085 |
| gpt_oss_20b | multihoprag | OracleComplete_PreRegisteredSufficiencySaturation |            424 |                     424 |                               0 |              417 |            98.3491 |                  153 |                36.0849 |                      8 |          0.287736  |        0.333218 |                                49.5283 |

**Paired answer flips**

| model       | dataset     | condition_a                    | condition_b                                       |   n_pairs |   n_technical_excluded |   both_correct |   control_correct_attack_wrong |   control_wrong_attack_correct |   both_wrong |   net_harm |   net_harm_pct_points |
|:------------|:------------|:-------------------------------|:--------------------------------------------------|----------:|-----------------------:|---------------:|-------------------------------:|-------------------------------:|-------------:|-----------:|----------------------:|
| gpt_oss_20b | hotpot      | NaturalClean                   | NaturalNeutral                                    |       378 |                      0 |            109 |                             37 |                             31 |          201 |          6 |              1.5873   |
| gpt_oss_20b | hotpot      | NaturalNeutral                 | Natural_epistemic_confirmation                    |       372 |                      0 |            111 |                             30 |                             40 |          191 |        -10 |             -2.68817  |
| gpt_oss_20b | hotpot      | NaturalNeutral                 | Natural_search_match_cue                          |       352 |                      0 |             96 |                             40 |                             33 |          183 |          7 |              1.98864  |
| gpt_oss_20b | hotpot      | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       364 |                      0 |             94 |                             41 |                             38 |          191 |          3 |              0.824176 |
| gpt_oss_20b | hotpot      | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       362 |                      0 |            102 |                             35 |                             49 |          176 |        -14 |             -3.8674   |
| gpt_oss_20b | hotpot      | NaturalClean                   | Natural_epistemic_confirmation                    |       411 |                      0 |            115 |                             43 |                             50 |          203 |         -7 |             -1.70316  |
| gpt_oss_20b | hotpot      | NaturalClean                   | Natural_search_match_cue                          |       388 |                      0 |            108 |                             37 |                             32 |          211 |          5 |              1.28866  |
| gpt_oss_20b | hotpot      | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       405 |                      0 |            109 |                             44 |                             38 |          214 |          6 |              1.48148  |
| gpt_oss_20b | hotpot      | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       403 |                      0 |            111 |                             44 |                             51 |          197 |         -7 |             -1.73697  |
| gpt_oss_20b | hotpot      | NaturalNeutral                 | NaturalContinueSearch                             |       372 |                      0 |            103 |                             40 |                             33 |          196 |          7 |              1.88172  |
| gpt_oss_20b | hotpot      | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       406 |                      0 |            126 |                             38 |                             49 |          193 |        -11 |             -2.70936  |
| gpt_oss_20b | hotpot      | NaturalNeutral                 | Natural_source_coverage                           |       367 |                      0 |            103 |                             37 |                             44 |          183 |         -7 |             -1.90736  |
| gpt_oss_20b | hotpot      | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       407 |                      0 |            132 |                             35 |                             36 |          204 |         -1 |             -0.2457   |
| gpt_oss_20b | hotpot      | Natural_search_match_cue       | Natural_search_match_source                       |       376 |                      0 |             97 |                             39 |                             43 |          197 |         -4 |             -1.06383  |
| gpt_oss_20b | 2wiki       | NaturalClean                   | NaturalNeutral                                    |       450 |                      0 |            113 |                             48 |                             40 |          249 |          8 |              1.77778  |
| gpt_oss_20b | 2wiki       | NaturalNeutral                 | Natural_epistemic_confirmation                    |       450 |                      0 |            112 |                             41 |                             49 |          248 |         -8 |             -1.77778  |
| gpt_oss_20b | 2wiki       | NaturalNeutral                 | Natural_search_match_cue                          |       450 |                      0 |            108 |                             45 |                             48 |          249 |         -3 |             -0.666667 |
| gpt_oss_20b | 2wiki       | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       450 |                      0 |            115 |                             38 |                             50 |          247 |        -12 |             -2.66667  |
| gpt_oss_20b | 2wiki       | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |            119 |                             34 |                             46 |          251 |        -12 |             -2.66667  |
| gpt_oss_20b | 2wiki       | NaturalClean                   | Natural_epistemic_confirmation                    |       450 |                      0 |            119 |                             42 |                             42 |          247 |          0 |              0        |
| gpt_oss_20b | 2wiki       | NaturalClean                   | Natural_search_match_cue                          |       450 |                      0 |            110 |                             51 |                             46 |          243 |          5 |              1.11111  |
| gpt_oss_20b | 2wiki       | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       450 |                      0 |            112 |                             49 |                             53 |          236 |         -4 |             -0.888889 |
| gpt_oss_20b | 2wiki       | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       450 |                      0 |            116 |                             45 |                             49 |          240 |         -4 |             -0.888889 |
| gpt_oss_20b | 2wiki       | NaturalNeutral                 | NaturalContinueSearch                             |       450 |                      0 |            109 |                             44 |                             45 |          252 |         -1 |             -0.222222 |
| gpt_oss_20b | 2wiki       | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       450 |                      0 |            134 |                             39 |                             45 |          232 |         -6 |             -1.33333  |
| gpt_oss_20b | 2wiki       | NaturalNeutral                 | Natural_source_coverage                           |       400 |                      0 |            108 |                             32 |                             43 |          217 |        -11 |             -2.75     |
| gpt_oss_20b | 2wiki       | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       450 |                      0 |            113 |                             48 |                             43 |          246 |          5 |              1.11111  |
| gpt_oss_20b | 2wiki       | Natural_search_match_cue       | Natural_search_match_source                       |       450 |                      0 |            103 |                             53 |                             40 |          254 |         13 |              2.88889  |
| gpt_oss_20b | musique     | NaturalClean                   | NaturalNeutral                                    |       406 |                      0 |             35 |                             38 |                             36 |          297 |          2 |              0.492611 |
| gpt_oss_20b | musique     | NaturalNeutral                 | Natural_epistemic_confirmation                    |       407 |                      0 |             40 |                             31 |                             35 |          301 |         -4 |             -0.982801 |
| gpt_oss_20b | musique     | NaturalNeutral                 | Natural_search_match_cue                          |       403 |                      0 |             33 |                             33 |                             27 |          310 |          6 |              1.48883  |
| gpt_oss_20b | musique     | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       411 |                      0 |             30 |                             38 |                             26 |          317 |         12 |              2.91971  |
| gpt_oss_20b | musique     | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       411 |                      0 |             32 |                             35 |                             27 |          317 |          8 |              1.94647  |
| gpt_oss_20b | musique     | NaturalClean                   | Natural_epistemic_confirmation                    |       405 |                      0 |             44 |                             30 |                             32 |          299 |         -2 |             -0.493827 |
| gpt_oss_20b | musique     | NaturalClean                   | Natural_search_match_cue                          |       399 |                      0 |             33 |                             36 |                             30 |          300 |          6 |              1.50376  |
| gpt_oss_20b | musique     | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       407 |                      0 |             35 |                             39 |                             22 |          311 |         17 |              4.1769   |
| gpt_oss_20b | musique     | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       410 |                      0 |             35 |                             37 |                             22 |          316 |         15 |              3.65854  |
| gpt_oss_20b | musique     | NaturalNeutral                 | NaturalContinueSearch                             |       407 |                      0 |             36 |                             34 |                             28 |          309 |          6 |              1.4742   |
| gpt_oss_20b | musique     | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       411 |                      0 |             36 |                             30 |                             33 |          312 |         -3 |             -0.729927 |
| gpt_oss_20b | musique     | NaturalNeutral                 | Natural_source_coverage                           |       402 |                      0 |             33 |                             33 |                             20 |          316 |         13 |              3.23383  |
| gpt_oss_20b | musique     | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       408 |                      0 |             38 |                             37 |                             22 |          311 |         15 |              3.67647  |
| gpt_oss_20b | musique     | Natural_search_match_cue       | Natural_search_match_source                       |       408 |                      0 |             37 |                             28 |                             29 |          314 |         -1 |             -0.245098 |
| gpt_oss_20b | multihoprag | NaturalClean                   | NaturalNeutral                                    |       380 |                      0 |             90 |                             48 |                             49 |          193 |         -1 |             -0.263158 |
| gpt_oss_20b | multihoprag | NaturalNeutral                 | Natural_epistemic_confirmation                    |       385 |                      0 |             99 |                             45 |                             46 |          195 |         -1 |             -0.25974  |
| gpt_oss_20b | multihoprag | NaturalNeutral                 | Natural_search_match_cue                          |       392 |                      0 |             95 |                             47 |                             50 |          200 |         -3 |             -0.765306 |
| gpt_oss_20b | multihoprag | NaturalNeutral                 | Natural_PreRegisteredSufficiency                  |       394 |                      0 |             96 |                             46 |                             51 |          201 |         -5 |             -1.26904  |
| gpt_oss_20b | multihoprag | NaturalNeutral                 | Natural_PreRegisteredSufficiencySaturation        |       392 |                      0 |             92 |                             51 |                             52 |          197 |         -1 |             -0.255102 |
| gpt_oss_20b | multihoprag | NaturalClean                   | Natural_epistemic_confirmation                    |       377 |                      0 |             93 |                             47 |                             44 |          193 |          3 |              0.795756 |
| gpt_oss_20b | multihoprag | NaturalClean                   | Natural_search_match_cue                          |       382 |                      0 |             97 |                             39 |                             40 |          206 |         -1 |             -0.26178  |
| gpt_oss_20b | multihoprag | NaturalClean                   | Natural_PreRegisteredSufficiency                  |       387 |                      0 |             92 |                             48 |                             48 |          199 |          0 |              0        |
| gpt_oss_20b | multihoprag | NaturalClean                   | Natural_PreRegisteredSufficiencySaturation        |       387 |                      0 |             92 |                             49 |                             47 |          199 |          2 |              0.516796 |
| gpt_oss_20b | multihoprag | NaturalNeutral                 | NaturalContinueSearch                             |       391 |                      0 |             99 |                             42 |                             40 |          210 |          2 |              0.511509 |
| gpt_oss_20b | multihoprag | OracleComplete_Neutral         | OracleComplete_PreRegisteredSufficiencySaturation |       403 |                      0 |             95 |                             56 |                             50 |          202 |          6 |              1.48883  |
| gpt_oss_20b | multihoprag | NaturalNeutral                 | Natural_source_coverage                           |       389 |                      0 |             96 |                             44 |                             53 |          196 |         -9 |             -2.31362  |
| gpt_oss_20b | multihoprag | Natural_epistemic_confirmation | Natural_epistemic_confirmation_source             |       383 |                      0 |             98 |                             45 |                             53 |          187 |         -8 |             -2.08877  |
| gpt_oss_20b | multihoprag | Natural_search_match_cue       | Natural_search_match_source                       |       388 |                      0 |            109 |                             35 |                             39 |          205 |         -4 |             -1.03093  |

### Retrieval-equivalence sensitivity summary [table: data/retrieval_equivalence.csv]

Main result is the same-example end-to-end comparison (`n_readable_pairs` denominator); `n_identical_observation_sensitivity_population` is the stricter subset where retrieval was byte-identical before any cue suffix, with its own denominator (spec 5.6) -- see [table: data/retrieval_equivalence.csv] for per-pair detail.

| model       | comparison                                                                  |   n_readable_pairs |   pct_identical_doc_set |   n_identical_observation_sensitivity_population |   n_total_pairs |
|:------------|:----------------------------------------------------------------------------|-------------------:|------------------------:|-------------------------------------------------:|----------------:|
| gpt_oss_20b | NaturalClean vs NaturalNeutral                                              |               1614 |                 62.7633 |                                              394 |            1761 |
| gpt_oss_20b | NaturalClean vs Natural_PreRegisteredSufficiency                            |               1649 |                 53.305  |                                              151 |            1800 |
| gpt_oss_20b | NaturalClean vs Natural_PreRegisteredSufficiencySaturation                  |               1650 |                 52.8485 |                                              148 |            1800 |
| gpt_oss_20b | NaturalClean vs Natural_epistemic_confirmation                              |               1643 |                 51.6129 |                                              140 |            1797 |
| gpt_oss_20b | NaturalClean vs Natural_search_match_cue                                    |               1619 |                 53.3045 |                                              143 |            1783 |
| gpt_oss_20b | NaturalNeutral vs NaturalContinueSearch                                     |               1620 |                 57.284  |                                              246 |            1761 |
| gpt_oss_20b | NaturalNeutral vs Natural_PreRegisteredSufficiency                          |               1619 |                 53.4898 |                                              134 |            1761 |
| gpt_oss_20b | NaturalNeutral vs Natural_PreRegisteredSufficiencySaturation                |               1615 |                 53.0031 |                                              143 |            1761 |
| gpt_oss_20b | NaturalNeutral vs Natural_epistemic_confirmation                            |               1614 |                 52.8501 |                                              136 |            1758 |
| gpt_oss_20b | NaturalNeutral vs Natural_search_match_cue                                  |               1597 |                 53.6631 |                                              147 |            1744 |
| gpt_oss_20b | NaturalNeutral vs Natural_source_coverage                                   |               1558 |                 52.6958 |                                              142 |            1711 |
| gpt_oss_20b | Natural_epistemic_confirmation vs Natural_epistemic_confirmation_source     |               1648 |                 53.1553 |                                              177 |            1797 |
| gpt_oss_20b | Natural_search_match_cue vs Natural_search_match_source                     |               1622 |                 54.1307 |                                              156 |            1783 |
| gpt_oss_20b | OracleComplete_Neutral vs OracleComplete_PreRegisteredSufficiencySaturation |               1670 |                100      |                                              464 |            1800 |

### HotpotQA -- narrative summary

Valid runs: 5015/5400. Three shards did not finish and about 300 runs failed with the vLLM Harmony parsing crash; all are reported as failures [table: data/run_inventory.csv]. Neutral PCR@1 = 6.9%, PCR@5 = 60.0% [table: data/pcr_stepwise.csv].

Route mix: roughly a quarter of gpt-oss episodes end with `ResearchComplete` and the rest by a no-tool exit, spread over about ten steps [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv], so effects remain visible at PCR@5.

Sufficiency +8.8 pp [+4.4, +12.9] at PCR@1 and +12.9 pp [+7.1, +19.2] at PCR@5; Saturation +16.9 pp [+11.9, +22.2] and +10.8 pp [+5.0, +16.6]. Continue Search -2.7 pp [-5.7, +0.0] [table: data/pcr_paired_deltas_stepwise.csv].

Saturation cuts search calls after d1 from 3.40 to 2.35 [table: data/process_metrics.csv]; re-delegation is unchanged (44.5% vs 45.6%), so the Supervisor does not compensate.

Evidence loss -0.8 pp [-3.3, +1.7], not significant [table: data/evidence_summary.csv]. Accuracy 38.2% Clean vs 40.9% Saturation [table: data/answer_summary.csv]; net flips -3.9 pp (35 vs 49, n = 362): slightly *fewer* answers turned wrong [table: data/answer_paired_flips.csv].

See the retrieval-equivalence table above for identical-observation counts [table: data/retrieval_equivalence.csv].

Supported: earlier stopping and fewer searches. Not supported: harm to evidence or answers. Served without the lenient Harmony parser (see Scope notes). [table: data/pcr_paired_deltas_stepwise.csv]

### 2WikiMultiHopQA -- narrative summary

Valid runs: 5350/5400; one shard missing and no parser failures [table: data/run_inventory.csv]. **Served with the lenient Harmony parser, unlike the other gpt-oss datasets** (Scope notes). Neutral PCR@1 = 7.3%, PCR@5 = 75.1% [table: data/pcr_stepwise.csv].

Route mix: roughly a quarter of gpt-oss episodes end with `ResearchComplete` and the rest by a no-tool exit, spread over about ten steps [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv], so effects remain visible at PCR@5.

Sufficiency +13.8 pp [+9.8, +17.8] at PCR@1 (+2.4 pp at PCR@5, n.s.); Saturation +10.4 pp [+6.4, +14.4] and +6.2 pp [+1.6, +10.9] at PCR@5; Continue Search -5.1 pp [-7.8, -2.4] [table: data/pcr_paired_deltas_stepwise.csv].

Search calls after d1 2.97 → 2.30; re-delegation 51.3% → 50.4% [table: data/process_metrics.csv].

Evidence loss +2.7 pp [+0.7, +4.8], significant: the clearest gpt-oss evidence effect [table: data/evidence_summary.csv]. Accuracy 35.8% Clean vs 36.7% Saturation [table: data/answer_summary.csv]; net flips -2.7 pp (34 vs 46, n = 450) [table: data/answer_paired_flips.csv].

See the retrieval-equivalence table above [table: data/retrieval_equivalence.csv].

Supported: earlier stopping and a small evidence loss. Not supported: an accuracy loss. Compare with other gpt-oss datasets only with the parser difference in mind. [table: data/pcr_paired_deltas_stepwise.csv]

### MuSiQue -- narrative summary

Valid runs: 5129/5400; about 270 parser-crash failures [table: data/run_inventory.csv]. Neutral PCR@1 = 8.0%, PCR@5 = 68.6% [table: data/pcr_stepwise.csv].

Route mix: roughly a quarter of gpt-oss episodes end with `ResearchComplete` and the rest by a no-tool exit, spread over about ten steps [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv], so effects remain visible at PCR@5.

Sufficiency +8.8 pp [+4.9, +12.7] and Saturation +8.3 pp [+4.2, +12.3] at PCR@1; neither is significant at PCR@5. Continue Search -5.0 pp [-7.9, -2.2] [table: data/pcr_paired_deltas_stepwise.csv].

Search calls after d1 2.99 → 2.54; re-delegation 47.3% → 45.3% [table: data/process_metrics.csv].

Evidence loss +0.1 pp [-3.0, +3.1] [table: data/evidence_summary.csv]. Accuracy 17.7% Clean vs 14.1% Saturation [table: data/answer_summary.csv]; net flips +1.9 pp (35 vs 27, n = 411), the only gpt-oss dataset where the sign points to harm, and not established as significant [table: data/answer_paired_flips.csv].

See the retrieval-equivalence table above [table: data/retrieval_equivalence.csv].

Supported: earlier stopping at step 1. Not supported: reliable harm. [table: data/pcr_paired_deltas_stepwise.csv]

### MultiHopRAG -- narrative summary

Valid runs: 5017/5400; one shard unfinished and about 380 parser-crash failures [table: data/run_inventory.csv]. Neutral PCR@1 = 8.4%, PCR@5 = 61.7% [table: data/pcr_stepwise.csv].

Route mix: roughly a quarter of gpt-oss episodes end with `ResearchComplete` and the rest by a no-tool exit, spread over about ten steps [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv], so effects remain visible at PCR@5.

Sufficiency +6.1 pp [+2.0, +10.2] at PCR@1 and +7.9 pp [+2.0, +13.7] at PCR@5; Saturation +10.5 pp [+5.9, +15.1] and +12.5 pp [+7.4, +18.1]; Continue Search -3.3 pp [-6.6, +0.0] [table: data/pcr_paired_deltas_stepwise.csv].

Search calls after d1 3.25 → 2.46; re-delegation rises 65.1% → 69.8% [table: data/process_metrics.csv].

Evidence loss +0.1 pp [-0.4, +0.6] [table: data/evidence_summary.csv]. Accuracy 36.1% Clean vs 36.0% Saturation [table: data/answer_summary.csv]; net flips -0.3 pp (51 vs 52, n = 392) [table: data/answer_paired_flips.csv].

See the retrieval-equivalence table above [table: data/retrieval_equivalence.csv].

Supported: earlier stopping at steps 1 and 5. Not supported: harm. [table: data/pcr_paired_deltas_stepwise.csv]

## 7.4 Direct model comparison

Both models stop earlier under the pre-registered sufficiency cues in all four datasets [table: data/pcr_paired_deltas_stepwise.csv], but on different timescales. Gemma's whole stopping decision happens at steps 1-2: it never calls `ResearchComplete`, and every condition reaches PCR = 100% by step 5 [table: data/pcr_stepwise.csv] [table: data/route_decomposition_stepwise.csv]. The cue moves PCR@1 by +12 to +26 pp and nothing later can differ. gpt-oss stops gradually over about ten steps, mostly by no-tool exits, with a baseline PCR@1 of only 7-8%; its cue effects are smaller at PCR@1 (+6 to +17 pp) but persist to PCR@5 in two (Sufficiency) or three (Saturation) datasets.

Baselines differ: Neutral PCR@1 is 16-31% for Gemma and 7-8% for gpt-oss [table: data/pcr_stepwise.csv], so raw deltas are not directly comparable. gpt-oss searches far more after d1 (about 3 calls vs under 1 for Gemma) and the cue removes about 0.4-1.0 of those calls; Gemma loses about 0.2-0.3 [table: data/process_metrics.csv]. Technical failures are concentrated in gpt-oss hotpot/musique/multihoprag, served without the lenient Harmony parser [table: data/run_inventory.csv]. Neither model shows a consistent final-answer loss [table: data/answer_paired_flips.csv].

## 7.5 Dataset comparison

The sufficiency effect replicates in every dataset for both models [table: data/pcr_paired_deltas_stepwise.csv]. It is largest on MultiHop-RAG for Gemma (+23.6 to +26.2 pp at PCR@1) and smallest on MuSiQue for both models. MuSiQue also has the lowest accuracy in every condition (about 14-21%) [table: data/answer_summary.csv], which limits any measurable harm. Evidence loss is significant in only two cells: Gemma/MultiHop-RAG (+0.9 pp) and gpt-oss/2Wiki (+2.7 pp) [table: data/evidence_summary.csv]. gpt-oss 2Wiki is not directly comparable with the other gpt-oss datasets because of the parser difference (Scope notes).

## 7.6 Main findings and negative findings

### Main findings

1. The two pre-registered cues (Sufficiency and Sufficiency+Saturation) cause earlier stopping in all 8 model × dataset cells, with every PCR@1 CI excluding zero [table: data/pcr_paired_deltas_stepwise.csv].
2. Search Match and Epistemic Confirmation have smaller effects that replicate in 3/4 datasets for Gemma and 1/4 for gpt-oss; Source Coverage alone has essentially no effect [table: data/pcr_paired_deltas_stepwise.csv].
3. Continue Search moves stopping the opposite way (later) in most cells, so the paradigm responds to cue content in both directions [table: data/pcr_paired_deltas_stepwise.csv].
4. The oracle comparison (all gold evidence supplied) shows the saturation cue still raises PCR@1 (+28.6 pp Gemma, +10.5 pp gpt-oss): the cue acts on stopping regardless of whether the evidence is actually complete [table: data/pcr_paired_deltas_stepwise.csv].

### Negative findings

1. Earlier stopping does not translate into a consistent evidence loss: Saturation evidence loss is significant in only 2 of 8 cells, and at most +2.7 pp [table: data/evidence_summary.csv].
2. Final-answer accuracy is not reliably affected: net flips range from -3.9 to +1.9 pp with no consistent sign [table: data/answer_paired_flips.csv]. A process-level effect is not shown to be downstream harm.
   *Update 2026-10-19:* the paired analysis in `results/whitebox/_out_of_scope/answer_harm/` shows the mechanism behind this flat average. When the cue costs evidence at the final-report input, answers are worse by about 10 pp; when it gains evidence they are better by about as much. The sufficiency cues produce only a small net evidence loss, so the average effect is about +0.25 pp, below detection [table: answer_harm/mechanism.csv] [table: answer_harm/evidence_shift.csv].
3. Gemma's PCR curves hit a ceiling by step 5 in every condition, so PCR@5+ cannot show effects for Gemma [table: data/pcr_stepwise.csv].
4. gpt-oss carries elevated technical failures in three datasets (parser crashes), and one dataset was served with a different parser; robustness claims for gpt-oss must account for both [table: data/run_inventory.csv].


# Main chapter tables

Compact PCR@k tables (spec section 8) -- full step-1-29 values are in `full_pcr_tables.md`.

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

## Retrieval-equivalence sensitivity summary

Main result is the same-example end-to-end comparison (`n_readable_pairs` denominator); `n_identical_observation_sensitivity_population` is the stricter subset where retrieval was byte-identical before any cue suffix, with its own denominator (spec 5.6) -- see [table: data/retrieval_equivalence.csv] for per-pair detail.

| model       | comparison                                                                  |   n_readable_pairs |   pct_identical_doc_set |   n_identical_observation_sensitivity_population |   n_total_pairs |
|:------------|:----------------------------------------------------------------------------|-------------------:|------------------------:|-------------------------------------------------:|----------------:|
| gemma27b    | NaturalClean vs NaturalNeutral                                              |               1750 |                 81.0286 |                                              765 |            1750 |
| gemma27b    | NaturalClean vs Natural_PreRegisteredSufficiency                            |               1800 |                 70.6111 |                                              510 |            1800 |
| gemma27b    | NaturalClean vs Natural_PreRegisteredSufficiencySaturation                  |               1700 |                 71.4706 |                                              481 |            1700 |
| gemma27b    | NaturalClean vs Natural_epistemic_confirmation                              |               1800 |                 71.7222 |                                              516 |            1800 |
| gemma27b    | NaturalClean vs Natural_search_match_cue                                    |               1800 |                 72.0556 |                                              546 |            1800 |
| gemma27b    | NaturalNeutral vs NaturalContinueSearch                                     |               1750 |                 78.0571 |                                              692 |            1750 |
| gemma27b    | NaturalNeutral vs Natural_PreRegisteredSufficiency                          |               1750 |                 71.5429 |                                              490 |            1750 |
| gemma27b    | NaturalNeutral vs Natural_PreRegisteredSufficiencySaturation                |               1650 |                 72.1818 |                                              460 |            1650 |
| gemma27b    | NaturalNeutral vs Natural_epistemic_confirmation                            |               1750 |                 73.4857 |                                              532 |            1750 |
| gemma27b    | NaturalNeutral vs Natural_search_match_cue                                  |               1750 |                 73.9429 |                                              550 |            1750 |
| gemma27b    | NaturalNeutral vs Natural_source_coverage                                   |               1750 |                 73.9429 |                                              542 |            1750 |
| gemma27b    | Natural_epistemic_confirmation vs Natural_epistemic_confirmation_source     |               1800 |                 73.2778 |                                              556 |            1800 |
| gemma27b    | Natural_search_match_cue vs Natural_search_match_source                     |               1800 |                 73      |                                              551 |            1800 |
| gemma27b    | OracleComplete_Neutral vs OracleComplete_PreRegisteredSufficiencySaturation |               1800 |                100      |                                              793 |            1800 |
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


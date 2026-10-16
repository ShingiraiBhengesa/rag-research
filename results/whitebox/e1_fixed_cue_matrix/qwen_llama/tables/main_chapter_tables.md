# Main chapter tables

Compact PCR@k tables (spec section 8) -- full step-1-29 values are in `full_pcr_tables.md`.

## Qwen2.5-14B-Instruct
### HotpotQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 443 | 23.9 | 90.1 | 98.9 | 100.0 |
| Neutral | 444 | 25.9 | 89.9 | 99.8 | 100.0 |
| ContinueSearch | 444 | 7.4 | 86.9 | 98.6 | 100.0 |
| Epistemic confirmation | 446 | 34.1 | 91.5 | 99.3 | 100.0 |
| Search-match cue | 446 | 43.7 | 92.6 | 99.8 | 100.0 |
| Source coverage | 444 | 30.2 | 90.1 | 99.5 | 100.0 |
| Epistemic confirmation + source | 446 | 34.3 | 90.4 | 99.3 | 100.0 |
| Search-match + source | 448 | 42.9 | 92.2 | 99.6 | 100.0 |
| PreRegisteredSufficiency | 446 | 55.8 | 93.9 | 100.0 | 100.0 |
| PreRegisteredSufficiencySaturation | 447 | 53.7 | 94.0 | 99.8 | 100.0 |
| Oracle Neutral | 443 | 25.7 | 86.9 | 99.1 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 446 | 54.7 | 95.1 | 100.0 | 100.0 |

### 2WikiMultiHopQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 444 | 31.1 | 92.8 | 100.0 | 100.0 |
| Neutral | 448 | 31.7 | 93.5 | 99.6 | 100.0 |
| ContinueSearch | 444 | 14.9 | 92.1 | 99.5 | 100.0 |
| Epistemic confirmation | 449 | 35.0 | 93.8 | 99.8 | 100.0 |
| Search-match cue | 447 | 39.6 | 95.3 | 99.6 | 100.0 |
| Source coverage | 446 | 31.4 | 94.2 | 99.8 | 100.0 |
| Epistemic confirmation + source | 446 | 37.4 | 94.4 | 99.8 | 100.0 |
| Search-match + source | 449 | 40.5 | 96.4 | 99.1 | 100.0 |
| PreRegisteredSufficiency | 449 | 53.2 | 94.7 | 99.3 | 100.0 |
| PreRegisteredSufficiencySaturation | 449 | 49.9 | 94.9 | 99.6 | 100.0 |
| Oracle Neutral | 450 | 40.0 | 92.0 | 99.1 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 449 | 69.5 | 95.3 | 98.9 | 100.0 |

### MuSiQue

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 441 | 14.1 | 78.9 | 98.2 | 100.0 |
| Neutral | 441 | 16.8 | 79.1 | 98.6 | 100.0 |
| ContinueSearch | 443 | 5.0 | 79.7 | 95.7 | 100.0 |
| Epistemic confirmation | 447 | 21.0 | 81.0 | 97.5 | 99.8 |
| Search-match cue | 444 | 25.5 | 81.5 | 98.4 | 100.0 |
| Source coverage | 442 | 16.3 | 79.2 | 98.2 | 100.0 |
| Epistemic confirmation + source | 438 | 16.9 | 80.6 | 97.7 | 100.0 |
| Search-match + source | 443 | 25.1 | 84.7 | 98.4 | 100.0 |
| PreRegisteredSufficiency | 444 | 41.7 | 89.2 | 97.7 | 100.0 |
| PreRegisteredSufficiencySaturation | 442 | 37.8 | 88.5 | 98.4 | 100.0 |
| Oracle Neutral | 440 | 14.5 | 83.2 | 97.0 | 99.8 |
| Oracle PreRegisteredSufficiencySaturation | 442 | 40.0 | 88.9 | 98.9 | 100.0 |

### MultiHopRAG

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 449 | 13.4 | 90.6 | 99.6 | 100.0 |
| Neutral | 450 | 14.7 | 89.3 | 100.0 | 100.0 |
| ContinueSearch | 450 | 1.1 | 85.3 | 99.8 | 100.0 |
| Epistemic confirmation | 450 | 20.0 | 89.3 | 99.8 | 100.0 |
| Search-match cue | 449 | 24.9 | 91.1 | 99.3 | 100.0 |
| Source coverage | 450 | 16.4 | 86.2 | 99.8 | 100.0 |
| Epistemic confirmation + source | 450 | 22.0 | 88.4 | 99.3 | 100.0 |
| Search-match + source | 450 | 21.8 | 90.4 | 99.8 | 100.0 |
| PreRegisteredSufficiency | 450 | 40.7 | 96.7 | 100.0 | 100.0 |
| PreRegisteredSufficiencySaturation | 450 | 42.9 | 96.2 | 99.6 | 100.0 |
| Oracle Neutral | 450 | 13.6 | 88.4 | 99.3 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 450 | 39.1 | 96.0 | 100.0 | 100.0 |

## Llama-3.1-8B-Instruct
### HotpotQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 428 | 3.7 | 74.3 | 96.5 | 99.8 |
| Neutral | 427 | 2.8 | 78.2 | 96.7 | 100.0 |
| ContinueSearch | 418 | 1.2 | 61.5 | 94.3 | 99.5 |
| Epistemic confirmation | 433 | 4.6 | 82.9 | 97.2 | 100.0 |
| Search-match cue | 434 | 2.1 | 84.3 | 97.5 | 100.0 |
| Source coverage | 433 | 7.4 | 79.0 | 97.9 | 100.0 |
| Epistemic confirmation + source | 436 | 4.4 | 86.7 | 97.5 | 100.0 |
| Search-match + source | 437 | 3.2 | 88.8 | 97.5 | 100.0 |
| PreRegisteredSufficiency | 438 | 4.3 | 89.7 | 97.5 | 100.0 |
| PreRegisteredSufficiencySaturation | 438 | 3.7 | 91.1 | 98.2 | 100.0 |
| Oracle Neutral | 436 | 2.5 | 78.0 | 97.5 | 99.8 |
| Oracle PreRegisteredSufficiencySaturation | 436 | 1.1 | 94.3 | 99.1 | 99.8 |

### 2WikiMultiHopQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 434 | 3.9 | 74.2 | 97.0 | 99.5 |
| Neutral | 437 | 2.1 | 73.7 | 97.3 | 100.0 |
| ContinueSearch | 428 | 0.5 | 47.7 | 92.5 | 99.3 |
| Epistemic confirmation | 439 | 3.6 | 80.0 | 97.7 | 99.8 |
| Search-match cue | 439 | 2.1 | 78.8 | 97.5 | 99.8 |
| Source coverage | 440 | 3.2 | 79.3 | 97.7 | 100.0 |
| Epistemic confirmation + source | 443 | 1.8 | 82.2 | 98.0 | 100.0 |
| Search-match + source | 443 | 0.9 | 81.5 | 97.5 | 100.0 |
| PreRegisteredSufficiency | 444 | 1.4 | 84.2 | 98.2 | 100.0 |
| PreRegisteredSufficiencySaturation | 444 | 1.1 | 85.8 | 98.6 | 100.0 |
| Oracle Neutral | 437 | 3.2 | 74.6 | 96.3 | 100.0 |
| Oracle PreRegisteredSufficiencySaturation | 444 | 1.8 | 86.5 | 98.4 | 100.0 |

### MuSiQue

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 428 | 5.4 | 72.9 | 98.6 | 99.3 |
| Neutral | 423 | 3.1 | 72.6 | 99.1 | 99.5 |
| ContinueSearch | 421 | 0.2 | 65.1 | 97.4 | 99.5 |
| Epistemic confirmation | 431 | 2.3 | 84.2 | 98.4 | 99.5 |
| Search-match cue | 432 | 0.5 | 81.2 | 98.4 | 99.8 |
| Source coverage | 422 | 4.3 | 83.4 | 98.1 | 99.8 |
| Epistemic confirmation + source | 429 | 2.1 | 88.1 | 97.7 | 100.0 |
| Search-match + source | 428 | 1.6 | 90.2 | 97.7 | 100.0 |
| PreRegisteredSufficiency | 430 | 1.9 | 94.0 | 98.6 | 100.0 |
| PreRegisteredSufficiencySaturation | 430 | 1.6 | 94.4 | 98.8 | 100.0 |
| Oracle Neutral | 414 | 2.7 | 74.9 | 96.6 | 99.8 |
| Oracle PreRegisteredSufficiencySaturation | 430 | 2.3 | 95.1 | 98.6 | 100.0 |

### MultiHopRAG

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 426 | 1.2 | 80.8 | 98.1 | 99.8 |
| Neutral | 428 | 0.7 | 78.5 | 99.1 | 100.0 |
| ContinueSearch | 423 | 0.2 | 76.1 | 99.1 | 100.0 |
| Epistemic confirmation | 429 | 1.6 | 89.7 | 99.3 | 100.0 |
| Search-match cue | 429 | 0.7 | 90.0 | 99.1 | 100.0 |
| Source coverage | 431 | 1.6 | 83.1 | 98.8 | 100.0 |
| Epistemic confirmation + source | 434 | 1.6 | 90.1 | 98.6 | 100.0 |
| Search-match + source | 435 | 0.9 | 94.5 | 98.6 | 100.0 |
| PreRegisteredSufficiency | 435 | 0.7 | 95.6 | 98.6 | 100.0 |
| PreRegisteredSufficiencySaturation | 435 | 0.7 | 97.0 | 98.6 | 100.0 |
| Oracle Neutral | 431 | 2.1 | 83.3 | 98.6 | 99.8 |
| Oracle PreRegisteredSufficiencySaturation | 433 | 0.5 | 96.8 | 98.8 | 100.0 |

## Retrieval-equivalence sensitivity summary

Main result is the same-example end-to-end comparison (`n_readable_pairs` denominator); `n_identical_observation_sensitivity_population` is the stricter subset where retrieval was byte-identical before any cue suffix, with its own denominator (spec 5.6) -- see `data/retrieval_equivalence.csv` for per-pair detail.

| model   | comparison                                                                  |   n_readable_pairs |   pct_identical_doc_set |   n_identical_observation_sensitivity_population |   n_total_pairs |
|:--------|:----------------------------------------------------------------------------|-------------------:|------------------------:|-------------------------------------------------:|----------------:|
| llama8b | NaturalClean vs NaturalNeutral                                              |               1728 |                 81.5394 |                                             1296 |            1800 |
| llama8b | NaturalClean vs Natural_PreRegisteredSufficiency                            |               1703 |                 46.9172 |                                              357 |            1800 |
| llama8b | NaturalClean vs Natural_PreRegisteredSufficiencySaturation                  |               1703 |                 46.9172 |                                              357 |            1800 |
| llama8b | NaturalClean vs Natural_epistemic_confirmation                              |               1728 |                 81.5394 |                                             1296 |            1800 |
| llama8b | NaturalClean vs Natural_search_match_cue                                    |               1728 |                 81.5394 |                                             1296 |            1800 |
| llama8b | NaturalNeutral vs NaturalContinueSearch                                     |               1744 |                 99.8853 |                                             1740 |            1800 |
| llama8b | NaturalNeutral vs Natural_PreRegisteredSufficiency                          |               1710 |                 46.1988 |                                              364 |            1800 |
| llama8b | NaturalNeutral vs Natural_PreRegisteredSufficiencySaturation                |               1710 |                 46.1988 |                                              364 |            1800 |
| llama8b | NaturalNeutral vs Natural_epistemic_confirmation                            |               1744 |                100      |                                             1744 |            1800 |
| llama8b | NaturalNeutral vs Natural_search_match_cue                                  |               1744 |                100      |                                             1744 |            1800 |
| llama8b | NaturalNeutral vs Natural_source_coverage                                   |               1726 |                 71.7845 |                                             1100 |            1800 |
| llama8b | Natural_epistemic_confirmation vs Natural_epistemic_confirmation_source     |               1710 |                 46.1988 |                                              364 |            1800 |
| llama8b | Natural_search_match_cue vs Natural_search_match_source                     |               1710 |                 46.1988 |                                              364 |            1800 |
| llama8b | OracleComplete_Neutral vs OracleComplete_PreRegisteredSufficiencySaturation |               1731 |                100      |                                             1066 |            1800 |
| qwen14b | NaturalClean vs NaturalNeutral                                              |               1799 |                 93.552  |                                             1487 |            1800 |
| qwen14b | NaturalClean vs Natural_PreRegisteredSufficiency                            |               1797 |                 54.1458 |                                              280 |            1800 |
| qwen14b | NaturalClean vs Natural_PreRegisteredSufficiencySaturation                  |               1797 |                 54.2571 |                                              277 |            1800 |
| qwen14b | NaturalClean vs Natural_epistemic_confirmation                              |               1800 |                 68.3889 |                                              712 |            1800 |
| qwen14b | NaturalClean vs Natural_search_match_cue                                    |               1798 |                 56.2848 |                                              341 |            1800 |
| qwen14b | NaturalNeutral vs NaturalContinueSearch                                     |               1797 |                 88.4808 |                                             1316 |            1800 |
| qwen14b | NaturalNeutral vs Natural_PreRegisteredSufficiency                          |               1796 |                 53.7862 |                                              280 |            1800 |
| qwen14b | NaturalNeutral vs Natural_PreRegisteredSufficiencySaturation                |               1796 |                 54.0089 |                                              265 |            1800 |
| qwen14b | NaturalNeutral vs Natural_epistemic_confirmation                            |               1799 |                 74.3747 |                                              985 |            1800 |
| qwen14b | NaturalNeutral vs Natural_search_match_cue                                  |               1797 |                 56.7613 |                                              457 |            1800 |
| qwen14b | NaturalNeutral vs Natural_source_coverage                                   |               1797 |                 54.0345 |                                              309 |            1800 |
| qwen14b | Natural_epistemic_confirmation vs Natural_epistemic_confirmation_source     |               1798 |                 60.2336 |                                              455 |            1800 |
| qwen14b | Natural_search_match_cue vs Natural_search_match_source                     |               1794 |                 65.8305 |                                              645 |            1800 |
| qwen14b | OracleComplete_Neutral vs OracleComplete_PreRegisteredSufficiencySaturation |               1795 |                 99.9443 |                                             1579 |            1800 |


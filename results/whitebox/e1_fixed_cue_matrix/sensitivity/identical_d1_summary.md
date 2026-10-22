# E1 identical-d1 sensitivity: summary at each model's characteristic step

ΔPCR in percentage points, cue − Neutral, 95% paired bootstrap CI. `identical_d1` = pairs whose pre-cue first search observation is byte-identical. Full curves for all 14 comparisons and steps 1–30: `identical_d1_pcr_deltas.csv`.

**Same sign as the all-paired estimate in 30/32 model × dataset × cue cells (both comparisons).**

## Fixed Sufficiency vs Neutral

| Model | Dataset | Step | All paired | Identical d1 | Same sign | Identical-d1 CI excludes 0 |
|---|---|---|---|---|---|---|
| qwen14b | hotpot | 1 | +29.7 [+24.7, +34.7] (n=441) | +18.2 [+7.6, +28.8] (n=66) | yes | yes |
| qwen14b | 2wiki | 1 | +21.7 [+16.3, +26.8] (n=447) | +25.4 [+14.1, +36.6] (n=71) | yes | yes |
| qwen14b | musique | 1 | +25.2 [+19.7, +30.7] (n=437) | +34.6 [+21.2, +48.1] (n=52) | yes | yes |
| qwen14b | multihoprag | 1 | +26.0 [+20.9, +31.1] (n=450) | +29.4 [+18.8, +40.0] (n=85) | yes | yes |
| llama8b | hotpot | 3 | +46.0 [+40.5, +51.4] (n=420) | +34.6 [+21.2, +48.1] (n=52) | yes | yes |
| llama8b | 2wiki | 3 | +32.8 [+27.5, +38.6] (n=433) | +19.8 [+11.0, +29.7] (n=91) | yes | yes |
| llama8b | musique | 3 | +53.3 [+47.9, +58.7] (n=407) | +41.2 [+29.4, +54.9] (n=51) | yes | yes |
| llama8b | multihoprag | 3 | +73.4 [+69.1, +78.2] (n=418) | +85.8 [+80.5, +91.1] (n=169) | yes | yes |
| gemma27b | hotpot | 1 | +17.6 [+12.2, +22.7] (n=450) | +22.1 [+12.5, +31.6] (n=136) | yes | yes |
| gemma27b | 2wiki | 1 | +20.0 [+15.6, +24.7] (n=450) | +18.0 [+11.9, +24.2] (n=194) | yes | yes |
| gemma27b | musique | 1 | +11.8 [+6.5, +16.8] (n=400) | +8.8 [+0.9, +16.8] (n=113) | yes | yes |
| gemma27b | multihoprag | 1 | +26.2 [+20.2, +32.0] (n=450) | +27.7 [+10.6, +44.7] (n=47) | yes | yes |
| gpt_oss_20b | hotpot | 2 | +14.0 [+8.5, +19.8] (n=364) | +36.0 [+12.0, +56.0] (n=25) | yes | yes |
| gpt_oss_20b | 2wiki | 2 | +13.8 [+8.4, +19.1] (n=450) | +6.1 [-9.1, +19.7] (n=66) | yes | no |
| gpt_oss_20b | musique | 2 | +9.0 [+4.4, +13.9] (n=409) | -3.7 [-22.2, +14.8] (n=27) | no | no |
| gpt_oss_20b | multihoprag | 2 | +11.2 [+6.6, +16.0] (n=394) | +6.2 [-12.5, +25.0] (n=16) | yes | no |

## Suff. + Saturation vs Neutral

| Model | Dataset | Step | All paired | Identical d1 | Same sign | Identical-d1 CI excludes 0 |
|---|---|---|---|---|---|---|
| qwen14b | hotpot | 1 | +27.7 [+22.7, +32.9] (n=441) | +15.2 [+4.5, +25.8] (n=66) | yes | yes |
| qwen14b | 2wiki | 1 | +18.3 [+12.8, +23.7] (n=447) | +23.9 [+12.7, +35.2] (n=71) | yes | yes |
| qwen14b | musique | 1 | +21.7 [+16.1, +26.7] (n=434) | +27.5 [+13.7, +41.2] (n=51) | yes | yes |
| qwen14b | multihoprag | 1 | +28.2 [+22.9, +34.0] (n=450) | +41.7 [+30.6, +52.8] (n=72) | yes | yes |
| llama8b | hotpot | 3 | +48.8 [+43.3, +54.1] (n=420) | +34.6 [+21.2, +48.1] (n=52) | yes | yes |
| llama8b | 2wiki | 3 | +36.7 [+31.2, +42.5] (n=433) | +22.0 [+12.1, +31.9] (n=91) | yes | yes |
| llama8b | musique | 3 | +59.2 [+54.1, +64.4] (n=407) | +49.0 [+35.3, +62.7] (n=51) | yes | yes |
| llama8b | multihoprag | 3 | +78.2 [+73.9, +82.5] (n=418) | +89.9 [+85.2, +94.1] (n=169) | yes | yes |
| gemma27b | hotpot | 1 | +19.8 [+14.2, +25.1] (n=450) | +20.3 [+12.7, +27.8] (n=158) | yes | yes |
| gemma27b | 2wiki | 1 | +14.6 [+9.4, +19.4] (n=350) | +12.2 [+5.4, +18.9] (n=148) | yes | yes |
| gemma27b | musique | 1 | +17.5 [+12.0, +22.5] (n=400) | +13.1 [+3.7, +22.4] (n=107) | yes | yes |
| gemma27b | multihoprag | 1 | +23.6 [+17.8, +29.3] (n=450) | +12.8 [-4.3, +29.8] (n=47) | yes | no |
| gpt_oss_20b | hotpot | 2 | +19.4 [+14.1, +25.2] (n=361) | +7.7 [-15.4, +30.8] (n=26) | yes | no |
| gpt_oss_20b | 2wiki | 2 | +14.9 [+9.5, +20.0] (n=450) | +25.0 [+10.9, +39.1] (n=64) | yes | yes |
| gpt_oss_20b | musique | 2 | +5.9 [+1.2, +10.8] (n=408) | -12.0 [-24.0, +0.0] (n=25) | no | no |
| gpt_oss_20b | multihoprag | 2 | +14.3 [+9.2, +19.6] (n=392) | +3.6 [-10.7, +17.9] (n=28) | yes | no |


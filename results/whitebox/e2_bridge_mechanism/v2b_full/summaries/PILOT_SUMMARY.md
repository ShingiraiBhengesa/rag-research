# Pilot summary (generated)

Runs are never truncated: every run reaches final_report_generation. PCR@30 = end of budget.

## PRIMARY (per cell: premature stop if the baseline gate PASSED, else stopped by step k), paired per question

| victim | dataset | contrast | tier | n | cue rate | control rate | diff [95% CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|
| qwen14b | 2wiki | structure [premature_stop_no_hop2_evidence] | primary | 125 | 0.136 | 0.104 | +3.2 [-3.2, 9.6] | 0.4545 |
| qwen14b | 2wiki | truth [premature_stop_no_hop2_evidence] | primary | 100 | 0.170 | 0.090 | +8.0 [-1.0, 16.0] | 0.1338 |
| qwen14b | 2wiki | selection [premature_stop_no_hop2_evidence] | primary | 136 | 0.125 | 0.125 | +0.0 [-5.1, 5.1] | 1 |
| qwen14b | 2wiki | bridge_k3_vs_fixed_sufficiency [premature_stop_no_hop2_evidence] | secondary | 136 | 0.125 | 0.147 | -2.2 [-8.1, 3.7] | 0.6476 |
| qwen14b | 2wiki | bridge_sat_k3_vs_fixed_saturation [premature_stop_no_hop2_evidence] | secondary | 113 | 0.150 | 0.159 | -0.9 [-8.8, 7.1] | 1 |
| qwen14b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 [premature_stop_no_hop2_evidence] | secondary | 113 | 0.150 | 0.124 | +2.6 [-4.4, 9.7] | 0.6291 |
| qwen14b | 2wiki | fixed_sufficiency_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 218 | 0.165 | 0.115 | +5.1 [-0.5, 10.1] | 0.08953 |
| qwen14b | 2wiki | fixed_saturation_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 218 | 0.156 | 0.115 | +4.1 [-1.4, 9.2] | 0.1628 |
| qwen14b | 2wiki | bridge_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 136 | 0.125 | 0.073 | +5.1 [-0.7, 11.0] | 0.1435 |
| qwen14b | 2wiki | bridge_k1_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 136 | 0.125 | 0.073 | +5.1 [-0.7, 11.0] | 0.1435 |
| qwen14b | 2wiki | entity_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 127 | 0.118 | 0.079 | +3.9 [-0.8, 8.7] | 0.1797 |
| qwen14b | 2wiki | unsupported_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 100 | 0.090 | 0.080 | +1.0 [-6.0, 8.0] | 1 |
| qwen14b | 2wiki | bridge_sat_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 113 | 0.150 | 0.062 | +8.8 [1.8, 16.8] | 0.03088 |
| qwen14b | hotpot | structure [premature_stop_no_hop2_evidence] | primary | 56 | 0.196 | 0.107 | +8.9 [0.0, 17.9] | 0.125 |
| qwen14b | hotpot | truth [premature_stop_no_hop2_evidence] | primary | 25 | 0.200 | 0.120 | +8.0 [0.0, 20.0] | 0.5 |
| qwen14b | hotpot | selection [premature_stop_no_hop2_evidence] | primary | 59 | 0.186 | 0.170 | +1.7 [-5.1, 8.5] | 1 |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency [premature_stop_no_hop2_evidence] | secondary | 59 | 0.186 | 0.170 | +1.7 [-6.8, 10.2] | 1 |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation [premature_stop_no_hop2_evidence] | secondary | 50 | 0.180 | 0.180 | +0.0 [-10.0, 10.0] | 1 |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 [premature_stop_no_hop2_evidence] | secondary | 49 | 0.163 | 0.184 | -2.0 [-12.2, 8.2] | 1 |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 135 | 0.296 | 0.215 | +8.2 [1.5, 14.8] | 0.0266 |
| qwen14b | hotpot | fixed_saturation_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 135 | 0.281 | 0.215 | +6.7 [0.0, 13.3] | 0.07835 |
| qwen14b | hotpot | bridge_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 59 | 0.186 | 0.051 | +13.6 [3.4, 23.7] | 0.02148 |
| qwen14b | hotpot | bridge_k1_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 59 | 0.170 | 0.051 | +11.9 [5.1, 20.3] | 0.01562 |
| qwen14b | hotpot | entity_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 63 | 0.127 | 0.079 | +4.8 [-3.2, 12.7] | 0.4531 |
| qwen14b | hotpot | unsupported_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 25 | 0.120 | 0.040 | +8.0 [0.0, 20.0] | 0.5 |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 50 | 0.180 | 0.060 | +12.0 [2.0, 24.0] | 0.07031 |
| llama8b | 2wiki | structure [stopped_by_step_2] | primary | 135 | 0.244 | 0.274 | -3.0 [-6.7, 0.7] | 0.2188 |
| llama8b | 2wiki | truth [stopped_by_step_2] | primary | 131 | 0.244 | 0.252 | -0.8 [-3.8, 2.3] | 1 |
| llama8b | 2wiki | selection [stopped_by_step_2] | primary | 167 | 0.239 | 0.264 | -2.4 [-5.4, 0.6] | 0.2891 |
| llama8b | 2wiki | bridge_k3_vs_fixed_sufficiency [stopped_by_step_2] | secondary | 166 | 0.241 | 0.277 | -3.6 [-6.6, -1.2] | 0.03125 |
| llama8b | 2wiki | bridge_sat_k3_vs_fixed_saturation [stopped_by_step_2] | secondary | 143 | 0.168 | 0.273 | -10.5 [-16.1, -5.6] | 6.1e-05 |
| llama8b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 [stopped_by_step_2] | secondary | 143 | 0.168 | 0.224 | -5.6 [-10.5, -1.4] | 0.03857 |
| llama8b | 2wiki | fixed_sufficiency_vs_neutral [stopped_by_step_2] | secondary | 314 | 0.213 | 0.073 | +14.0 [10.2, 18.1] | 0 |
| llama8b | 2wiki | fixed_saturation_vs_neutral [stopped_by_step_2] | secondary | 314 | 0.220 | 0.073 | +14.6 [10.8, 18.8] | 0 |
| llama8b | 2wiki | bridge_k3_vs_neutral [stopped_by_step_2] | secondary | 163 | 0.227 | 0.086 | +14.1 [8.0, 20.2] | 6e-06 |
| llama8b | 2wiki | bridge_k1_vs_neutral [stopped_by_step_2] | secondary | 163 | 0.252 | 0.086 | +16.6 [11.0, 22.7] | 0 |
| llama8b | 2wiki | entity_vs_neutral [stopped_by_step_2] | secondary | 140 | 0.250 | 0.086 | +16.4 [10.0, 23.6] | 2e-06 |
| llama8b | 2wiki | unsupported_vs_neutral [stopped_by_step_2] | secondary | 127 | 0.236 | 0.095 | +14.2 [7.9, 20.5] | 4e-05 |
| llama8b | 2wiki | bridge_sat_k3_vs_neutral [stopped_by_step_2] | secondary | 140 | 0.157 | 0.093 | +6.4 [0.0, 12.1] | 0.06357 |
| llama8b | hotpot | structure [stopped_by_step_2] | primary | 137 | 0.474 | 0.482 | -0.7 [-5.1, 3.6] | 1 |
| llama8b | hotpot | truth [stopped_by_step_2] | primary | 67 | 0.478 | 0.463 | +1.5 [-3.0, 7.5] | 1 |
| llama8b | hotpot | selection [stopped_by_step_2] | primary | 147 | 0.463 | 0.456 | +0.7 [-1.4, 2.7] | 1 |
| llama8b | hotpot | bridge_k3_vs_fixed_sufficiency [stopped_by_step_2] | secondary | 147 | 0.463 | 0.497 | -3.4 [-7.5, 0.0] | 0.1797 |
| llama8b | hotpot | bridge_sat_k3_vs_fixed_saturation [stopped_by_step_2] | secondary | 125 | 0.456 | 0.528 | -7.2 [-12.0, -3.2] | 0.003906 |
| llama8b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 [stopped_by_step_2] | secondary | 125 | 0.456 | 0.488 | -3.2 [-8.8, 1.6] | 0.3438 |
| llama8b | hotpot | fixed_sufficiency_vs_neutral [stopped_by_step_2] | secondary | 316 | 0.386 | 0.108 | +27.9 [23.1, 32.9] | 0 |
| llama8b | hotpot | fixed_saturation_vs_neutral [stopped_by_step_2] | secondary | 316 | 0.399 | 0.108 | +29.1 [24.1, 34.2] | 0 |
| llama8b | hotpot | bridge_k3_vs_neutral [stopped_by_step_2] | secondary | 145 | 0.455 | 0.152 | +30.3 [22.1, 38.6] | 0 |
| llama8b | hotpot | bridge_k1_vs_neutral [stopped_by_step_2] | secondary | 145 | 0.448 | 0.152 | +29.7 [21.4, 37.9] | 0 |
| llama8b | hotpot | entity_vs_neutral [stopped_by_step_2] | secondary | 144 | 0.472 | 0.153 | +31.9 [23.6, 40.3] | 0 |
| llama8b | hotpot | unsupported_vs_neutral [stopped_by_step_2] | secondary | 65 | 0.462 | 0.169 | +29.2 [18.5, 41.5] | 2.1e-05 |
| llama8b | hotpot | bridge_sat_k3_vs_neutral [stopped_by_step_2] | secondary | 124 | 0.452 | 0.177 | +27.4 [19.4, 35.5] | 0 |

## Per arm: primary rate and secondary outcomes

| victim | dataset | arm | runs | primary n | premature stop | abstention | search calls after d1 |
|---|---|---|---|---|---|---|---|
| llama8b | 2wiki | bridge_saturation_k3 | 143 | 64 | 0.8594 | 0.0 | 0.58 |
| llama8b | 2wiki | bridge_sufficiency_k1 | 167 | 76 | 0.8026 | 0.0 | 0.515 |
| llama8b | 2wiki | bridge_sufficiency_k3 | 167 | 76 | 0.8158 | 0.0 | 0.509 |
| llama8b | 2wiki | entity_sufficiency | 143 | 70 | 0.8143 | 0.0 | 0.608 |
| llama8b | 2wiki | fixed_saturation | 319 | 193 | 0.8446 | 0.0 | 0.636 |
| llama8b | 2wiki | fixed_sufficiency | 319 | 193 | 0.8342 | 0.0 | 0.652 |
| llama8b | 2wiki | neutral | 319 | 192 | 0.7448 | 0.0 | 0.815 |
| llama8b | 2wiki | unsupported_sufficiency | 131 | 63 | 0.7937 | 0.0 | 0.71 |
| llama8b | hotpot | bridge_saturation_k3 | 127 | 51 | 0.8431 | 0.0 | 0.441 |
| llama8b | hotpot | bridge_sufficiency_k1 | 148 | 60 | 0.9333 | 0.0 | 0.324 |
| llama8b | hotpot | bridge_sufficiency_k3 | 148 | 60 | 0.9 | 0.0 | 0.358 |
| llama8b | hotpot | entity_sufficiency | 147 | 57 | 0.8947 | 0.0 | 0.49 |
| llama8b | hotpot | fixed_saturation | 322 | 162 | 0.8642 | 0.0 | 0.466 |
| llama8b | hotpot | fixed_sufficiency | 322 | 162 | 0.8395 | 0.0 | 0.503 |
| llama8b | hotpot | neutral | 322 | 162 | 0.6914 | 0.0 | 0.95 |
| llama8b | hotpot | unsupported_sufficiency | 67 | 32 | 0.9375 | 0.0 | 0.537 |
| qwen14b | 2wiki | bridge_saturation_k3 | 152 | 113 | 0.1504 | 0.0 | 2.362 |
| qwen14b | 2wiki | bridge_sufficiency_k1 | 177 | 136 | 0.125 | 0.0 | 2.367 |
| qwen14b | 2wiki | bridge_sufficiency_k3 | 177 | 136 | 0.125 | 0.0 | 2.215 |
| qwen14b | 2wiki | entity_sufficiency | 164 | 127 | 0.1181 | 0.0 | 2.5 |
| qwen14b | 2wiki | fixed_saturation | 278 | 218 | 0.156 | 0.0 | 2.763 |
| qwen14b | 2wiki | fixed_sufficiency | 278 | 218 | 0.1651 | 0.0 | 2.921 |
| qwen14b | 2wiki | neutral | 278 | 218 | 0.1147 | 0.0036 | 3.317 |
| qwen14b | 2wiki | unsupported_sufficiency | 131 | 100 | 0.09 | 0.0 | 2.504 |
| qwen14b | hotpot | bridge_saturation_k3 | 102 | 50 | 0.18 | 0.0098 | 2.598 |
| qwen14b | hotpot | bridge_sufficiency_k1 | 124 | 59 | 0.1695 | 0.0081 | 3.476 |
| qwen14b | hotpot | bridge_sufficiency_k3 | 124 | 59 | 0.1864 | 0.0081 | 3.387 |
| qwen14b | hotpot | entity_sufficiency | 127 | 63 | 0.127 | 0.0157 | 3.528 |
| qwen14b | hotpot | fixed_saturation | 247 | 135 | 0.2815 | 0.004 | 3.073 |
| qwen14b | hotpot | fixed_sufficiency | 247 | 135 | 0.2963 | 0.004 | 3.457 |
| qwen14b | hotpot | neutral | 247 | 135 | 0.2148 | 0.0081 | 4.036 |
| qwen14b | hotpot | unsupported_sufficiency | 57 | 25 | 0.12 | 0.0351 | 3.316 |

## PCR contrasts (all paired questions; Δ in pp, 95% paired bootstrap CI)

| victim | dataset | contrast | n | Δ@1 | Δ@2 | Δ@5 | Δ@30 |
|---|---|---|---|---|---|---|---|
| llama8b | 2wiki | bridge_k1_vs_neutral | 163 | -1.2 [-3.1, 0.0] | +16.6 [11.0, 22.7] | +6.8 [1.2, 12.3] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_k3_vs_fixed_sufficiency | 166 | +0.0 [0.0, 0.0] | -3.6 [-6.6, -1.2] | -3.6 [-7.2, 0.0] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_k3_vs_neutral | 163 | -1.2 [-3.1, 0.0] | +14.1 [8.0, 20.2] | +6.8 [1.8, 11.7] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 143 | +0.0 [0.0, 0.0] | -5.6 [-10.5, -1.4] | +2.8 [-0.7, 6.3] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 143 | -0.7 [-2.1, 0.0] | -10.5 [-16.1, -5.6] | -4.2 [-7.7, -1.4] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_neutral | 140 | -0.7 [-2.1, 0.0] | +6.4 [0.0, 12.1] | +8.6 [3.6, 14.3] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | entity_vs_neutral | 140 | -1.4 [-3.6, 0.0] | +16.4 [10.0, 23.6] | +9.3 [4.3, 15.0] | +0.7 [0.0, 2.1] |
| llama8b | 2wiki | fixed_saturation_vs_neutral | 314 | -1.0 [-2.5, 0.6] | +14.6 [10.8, 18.8] | +15.0 [11.2, 19.1] | +0.3 [0.0, 1.0] |
| llama8b | 2wiki | fixed_sufficiency_vs_neutral | 314 | -1.6 [-3.2, -0.3] | +14.0 [10.2, 18.1] | +13.1 [9.2, 16.9] | +0.3 [0.0, 1.0] |
| llama8b | 2wiki | selection | 167 | +0.0 [0.0, 0.0] | -2.4 [-5.4, 0.6] | +0.0 [-2.4, 2.4] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | structure | 135 | +0.0 [0.0, 0.0] | -3.0 [-6.7, 0.7] | -1.5 [-5.2, 2.2] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | truth | 131 | -0.8 [-2.3, 0.0] | -0.8 [-3.8, 2.3] | +0.0 [-2.3, 2.3] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | unsupported_vs_neutral | 127 | +0.0 [-2.4, 2.4] | +14.2 [7.9, 20.5] | +10.2 [4.7, 15.8] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_k1_vs_neutral | 145 | -5.5 [-9.7, -2.1] | +29.7 [21.4, 37.9] | +11.7 [5.5, 17.9] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_k3_vs_fixed_sufficiency | 147 | -1.4 [-3.4, 0.0] | -3.4 [-7.5, 0.0] | -1.4 [-5.4, 2.0] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_k3_vs_neutral | 145 | -5.5 [-9.7, -2.1] | +30.3 [22.1, 38.6] | +12.4 [6.2, 18.6] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 125 | +0.0 [-2.4, 2.4] | -3.2 [-8.8, 1.6] | -2.4 [-5.6, 0.8] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_sat_k3_vs_fixed_saturation | 125 | -0.8 [-2.4, 0.0] | -7.2 [-12.0, -3.2] | -2.4 [-6.4, 0.8] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_sat_k3_vs_neutral | 124 | -6.5 [-10.5, -2.4] | +27.4 [19.4, 35.5] | +12.9 [6.5, 20.2] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | entity_vs_neutral | 144 | -5.6 [-9.7, -2.1] | +31.9 [23.6, 40.3] | +12.5 [6.9, 18.8] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | fixed_saturation_vs_neutral | 316 | -3.8 [-6.0, -1.9] | +29.1 [24.1, 34.2] | +15.5 [11.4, 19.9] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | fixed_sufficiency_vs_neutral | 316 | -2.9 [-4.8, -1.3] | +27.9 [23.1, 32.9] | +13.3 [9.5, 17.7] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | selection | 147 | +0.0 [0.0, 0.0] | +0.7 [-1.4, 2.7] | +0.7 [-1.4, 3.4] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | structure | 137 | +0.0 [0.0, 0.0] | -0.7 [-5.1, 3.6] | -0.7 [-3.6, 2.9] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | truth | 67 | +0.0 [0.0, 0.0] | +1.5 [-3.0, 7.5] | -1.5 [-7.5, 3.0] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | unsupported_vs_neutral | 65 | -6.2 [-12.3, -1.5] | +29.2 [18.5, 41.5] | +12.3 [3.1, 21.5] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_k1_vs_neutral | 177 | +21.5 [15.2, 28.2] | +9.6 [4.0, 15.2] | +4.0 [1.7, 6.8] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_k3_vs_fixed_sufficiency | 177 | -4.5 [-11.3, 2.3] | -4.5 [-9.0, 0.6] | -1.7 [-4.0, 0.6] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_k3_vs_neutral | 177 | +23.2 [16.4, 30.5] | +11.3 [6.2, 17.0] | +3.4 [1.1, 6.2] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 151 | +4.0 [-2.6, 10.0] | +4.0 [-0.0, 8.6] | +1.3 [-1.3, 4.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 152 | +0.7 [-5.3, 5.9] | +0.7 [-4.6, 5.9] | +0.0 [-2.0, 2.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_neutral | 152 | +26.3 [19.1, 34.2] | +14.5 [8.6, 20.4] | +4.6 [1.3, 7.9] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | entity_vs_neutral | 164 | +13.4 [7.3, 20.1] | +4.9 [-0.6, 11.0] | +1.8 [-0.6, 4.3] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | fixed_saturation_vs_neutral | 278 | +21.6 [16.6, 26.6] | +15.5 [11.1, 20.5] | +5.4 [2.5, 8.6] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | fixed_sufficiency_vs_neutral | 278 | +23.4 [18.0, 28.8] | +16.9 [12.2, 21.9] | +6.8 [3.6, 10.1] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | selection | 177 | +1.7 [-2.3, 6.2] | +1.7 [-1.1, 4.5] | -0.6 [-1.7, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | structure | 160 | +10.6 [3.8, 17.5] | +4.4 [-0.6, 9.4] | +1.2 [-1.9, 5.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | truth | 131 | +13.0 [6.1, 19.9] | +2.3 [-3.0, 8.4] | -2.3 [-5.3, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | unsupported_vs_neutral | 131 | +12.2 [4.6, 19.9] | +7.6 [0.0, 16.0] | +6.1 [2.3, 10.7] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k1_vs_neutral | 124 | +20.2 [12.9, 28.2] | +16.9 [8.9, 25.8] | +8.1 [3.2, 13.7] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency | 124 | -11.3 [-19.4, -3.2] | -8.1 [-14.5, -2.4] | -2.4 [-6.5, 1.6] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_neutral | 124 | +21.0 [13.7, 29.0] | +17.7 [10.5, 25.8] | +5.7 [0.8, 11.3] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 101 | +4.0 [-2.0, 10.9] | -1.0 [-6.9, 5.0] | +0.0 [-3.0, 3.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation | 102 | -7.8 [-15.7, 0.0] | -2.0 [-7.8, 3.0] | +0.0 [-3.9, 3.9] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral | 102 | +25.5 [16.7, 34.3] | +18.6 [10.8, 27.5] | +3.9 [1.0, 7.8] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | entity_vs_neutral | 127 | +12.6 [5.5, 20.5] | +16.5 [10.2, 23.6] | +3.1 [-1.6, 7.9] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_saturation_vs_neutral | 247 | +32.0 [26.3, 38.5] | +23.9 [18.2, 30.0] | +6.9 [3.2, 10.9] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral | 247 | +32.4 [25.9, 38.5] | +22.3 [17.0, 28.3] | +6.9 [3.2, 10.9] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | selection | 124 | +0.8 [-5.7, 6.5] | +0.8 [-4.0, 5.7] | -2.4 [-5.7, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | structure | 118 | +8.5 [0.9, 16.1] | +1.7 [-4.2, 8.5] | +3.4 [0.0, 7.6] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | truth | 57 | +12.3 [3.5, 22.8] | +7.0 [-3.5, 17.5] | +0.0 [-5.3, 5.3] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | unsupported_vs_neutral | 57 | +8.8 [0.0, 19.3] | +10.5 [0.0, 22.8] | +3.5 [0.0, 8.8] | +0.0 [0.0, 0.0] |

## Final-answer accuracy and evidence coverage by arm

| victim | dataset | arm | n | accuracy | mean evidence coverage | full coverage |
|---|---|---|---|---|---|---|
| llama8b | 2wiki | bridge_saturation_k3 | 143 | 0.3217 | 0.8182 | 0.6084 |
| llama8b | 2wiki | bridge_sufficiency_k1 | 167 | 0.3293 | 0.8234 | 0.5988 |
| llama8b | 2wiki | bridge_sufficiency_k3 | 167 | 0.2934 | 0.8234 | 0.6048 |
| llama8b | 2wiki | entity_sufficiency | 143 | 0.2937 | 0.8182 | 0.5734 |
| llama8b | 2wiki | fixed_saturation | 319 | 0.2508 | 0.7045 | 0.4639 |
| llama8b | 2wiki | fixed_sufficiency | 319 | 0.2571 | 0.7092 | 0.4671 |
| llama8b | 2wiki | neutral | 319 | 0.2571 | 0.7273 | 0.4953 |
| llama8b | 2wiki | unsupported_sufficiency | 131 | 0.2519 | 0.8206 | 0.5878 |
| llama8b | hotpot | bridge_saturation_k3 | 127 | 0.3228 | 0.8268 | 0.6614 |
| llama8b | hotpot | bridge_sufficiency_k1 | 148 | 0.3176 | 0.8074 | 0.6216 |
| llama8b | hotpot | bridge_sufficiency_k3 | 148 | 0.3108 | 0.8142 | 0.6351 |
| llama8b | hotpot | entity_sufficiency | 147 | 0.2857 | 0.8197 | 0.6463 |
| llama8b | hotpot | fixed_saturation | 322 | 0.2733 | 0.7453 | 0.5528 |
| llama8b | hotpot | fixed_sufficiency | 322 | 0.2919 | 0.7562 | 0.5683 |
| llama8b | hotpot | neutral | 322 | 0.2795 | 0.795 | 0.6429 |
| llama8b | hotpot | unsupported_sufficiency | 67 | 0.2985 | 0.7761 | 0.5522 |
| qwen14b | 2wiki | bridge_saturation_k3 | 152 | 0.4474 | 0.9326 | 0.8355 |
| qwen14b | 2wiki | bridge_sufficiency_k1 | 177 | 0.4689 | 0.9435 | 0.8531 |
| qwen14b | 2wiki | bridge_sufficiency_k3 | 177 | 0.5028 | 0.9492 | 0.8701 |
| qwen14b | 2wiki | entity_sufficiency | 164 | 0.4878 | 0.9451 | 0.878 |
| qwen14b | 2wiki | fixed_saturation | 278 | 0.4281 | 0.9182 | 0.8129 |
| qwen14b | 2wiki | fixed_sufficiency | 278 | 0.4353 | 0.92 | 0.8165 |
| qwen14b | 2wiki | neutral | 278 | 0.4209 | 0.929 | 0.8381 |
| qwen14b | 2wiki | unsupported_sufficiency | 131 | 0.4504 | 0.9504 | 0.8931 |
| qwen14b | hotpot | bridge_saturation_k3 | 102 | 0.4804 | 0.9559 | 0.9118 |
| qwen14b | hotpot | bridge_sufficiency_k1 | 124 | 0.4677 | 0.9556 | 0.9113 |
| qwen14b | hotpot | bridge_sufficiency_k3 | 124 | 0.4435 | 0.9516 | 0.9032 |
| qwen14b | hotpot | entity_sufficiency | 127 | 0.4016 | 0.9685 | 0.937 |
| qwen14b | hotpot | fixed_saturation | 247 | 0.413 | 0.917 | 0.8421 |
| qwen14b | hotpot | fixed_sufficiency | 247 | 0.4494 | 0.9109 | 0.8259 |
| qwen14b | hotpot | neutral | 247 | 0.4008 | 0.9332 | 0.8704 |
| qwen14b | hotpot | unsupported_sufficiency | 57 | 0.4561 | 0.9737 | 0.9474 |

## Paired answer flips and evidence loss per contrast

| victim | dataset | contrast | n | ctl✓→cue✗ | ctl✗→cue✓ | Δacc [CI] | evidence loss [CI] |
|---|---|---|---|---|---|---|---|
| llama8b | 2wiki | structure | 135 | 10 | 7 | -2.2 [-8.9, 3.7] | +1.1 [-0.4, 2.8] |
| llama8b | 2wiki | truth | 131 | 6 | 10 | +3.0 [-3.0, 9.2] | +0.8 [-0.9, 2.7] |
| llama8b | 2wiki | selection | 167 | 7 | 1 | -3.6 [-7.2, -0.6] | +0.0 [-0.8, 0.6] |
| llama8b | 2wiki | bridge_k3_vs_fixed_sufficiency | 167 | 9 | 9 | +0.0 [-4.8, 5.4] | +0.3 [-1.2, 1.8] |
| llama8b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 143 | 10 | 13 | +2.1 [-4.9, 8.4] | +0.4 [-1.1, 1.8] |
| llama8b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 143 | 6 | 10 | +2.8 [-2.8, 8.4] | +0.2 [-1.2, 1.6] |
| llama8b | 2wiki | fixed_sufficiency_vs_neutral | 319 | 15 | 15 | +0.0 [-3.1, 3.5] | +1.8 [0.2, 3.4] |
| llama8b | 2wiki | fixed_saturation_vs_neutral | 319 | 18 | 16 | -0.6 [-4.1, 2.8] | +2.3 [0.7, 3.9] |
| llama8b | 2wiki | bridge_k3_vs_neutral | 167 | 15 | 12 | -1.8 [-7.8, 4.2] | +2.1 [0.3, 4.0] |
| llama8b | 2wiki | bridge_k1_vs_neutral | 167 | 12 | 15 | +1.8 [-4.2, 8.4] | +2.1 [0.4, 3.9] |
| llama8b | 2wiki | entity_vs_neutral | 143 | 8 | 7 | -0.7 [-5.6, 4.9] | +1.9 [-0.2, 4.2] |
| llama8b | 2wiki | unsupported_vs_neutral | 131 | 16 | 8 | -6.1 [-13.0, 0.8] | +1.9 [-0.4, 4.2] |
| llama8b | 2wiki | bridge_sat_k3_vs_neutral | 143 | 13 | 11 | -1.4 [-7.7, 5.6] | +2.3 [0.7, 4.0] |
| llama8b | hotpot | structure | 138 | 7 | 9 | +1.5 [-4.3, 6.5] | -0.7 [-2.5, 1.1] |
| llama8b | hotpot | truth | 67 | 1 | 3 | +3.0 [-3.0, 9.0] | -0.8 [-3.0, 0.0] |
| llama8b | hotpot | selection | 148 | 4 | 3 | -0.7 [-4.0, 2.7] | -0.7 [-1.7, 0.0] |
| llama8b | hotpot | bridge_k3_vs_fixed_sufficiency | 148 | 12 | 8 | -2.7 [-8.8, 3.4] | +0.0 [-2.4, 2.4] |
| llama8b | hotpot | bridge_sat_k3_vs_fixed_saturation | 127 | 7 | 11 | +3.1 [-3.1, 9.4] | -1.6 [-4.7, 1.6] |
| llama8b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 127 | 4 | 6 | +1.6 [-3.1, 6.3] | -1.2 [-2.8, 0.0] |
| llama8b | hotpot | fixed_sufficiency_vs_neutral | 322 | 17 | 21 | +1.2 [-2.5, 5.0] | +3.9 [1.9, 5.9] |
| llama8b | hotpot | fixed_saturation_vs_neutral | 322 | 19 | 17 | -0.6 [-4.7, 2.8] | +5.0 [2.8, 7.1] |
| llama8b | hotpot | bridge_k3_vs_neutral | 148 | 10 | 7 | -2.0 [-7.4, 3.4] | +3.4 [1.4, 5.7] |
| llama8b | hotpot | bridge_k1_vs_neutral | 148 | 10 | 8 | -1.4 [-6.8, 4.0] | +4.0 [1.7, 6.4] |
| llama8b | hotpot | entity_vs_neutral | 147 | 12 | 7 | -3.4 [-9.5, 2.0] | +4.1 [1.7, 6.8] |
| llama8b | hotpot | unsupported_vs_neutral | 67 | 4 | 1 | -4.5 [-11.9, 1.5] | +3.0 [0.8, 6.0] |
| llama8b | hotpot | bridge_sat_k3_vs_neutral | 127 | 8 | 7 | -0.8 [-6.3, 5.5] | +2.4 [0.4, 4.7] |
| qwen14b | 2wiki | structure | 160 | 8 | 9 | +0.6 [-4.4, 5.6] | +0.6 [-1.9, 3.1] |
| qwen14b | 2wiki | truth | 131 | 8 | 13 | +3.8 [-3.0, 10.7] | +1.9 [-1.7, 5.3] |
| qwen14b | 2wiki | selection | 177 | 2 | 8 | +3.4 [0.0, 7.3] | -0.6 [-2.7, 1.6] |
| qwen14b | 2wiki | bridge_k3_vs_fixed_sufficiency | 177 | 9 | 13 | +2.3 [-3.4, 7.3] | -1.6 [-3.8, 1.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 152 | 11 | 8 | -2.0 [-7.9, 4.0] | -0.3 [-3.1, 2.5] |
| qwen14b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 151 | 14 | 6 | -5.3 [-10.6, 0.7] | +1.7 [-1.2, 4.3] |
| qwen14b | 2wiki | fixed_sufficiency_vs_neutral | 278 | 16 | 20 | +1.4 [-2.9, 5.8] | +0.9 [-1.4, 3.1] |
| qwen14b | 2wiki | fixed_saturation_vs_neutral | 278 | 15 | 17 | +0.7 [-3.2, 4.7] | +1.1 [-1.1, 3.2] |
| qwen14b | 2wiki | bridge_k3_vs_neutral | 177 | 6 | 11 | +2.8 [-1.7, 7.3] | +0.1 [-2.5, 2.5] |
| qwen14b | 2wiki | bridge_k1_vs_neutral | 177 | 9 | 8 | -0.6 [-5.1, 4.0] | +0.7 [-1.6, 3.1] |
| qwen14b | 2wiki | entity_vs_neutral | 164 | 5 | 10 | +3.0 [-1.2, 7.3] | +0.1 [-1.8, 2.1] |
| qwen14b | 2wiki | unsupported_vs_neutral | 131 | 9 | 10 | +0.8 [-5.3, 6.9] | -0.8 [-3.2, 1.9] |
| qwen14b | 2wiki | bridge_sat_k3_vs_neutral | 152 | 12 | 8 | -2.6 [-8.6, 3.3] | +2.0 [-0.8, 4.9] |
| qwen14b | hotpot | structure | 118 | 7 | 8 | +0.9 [-5.1, 7.6] | +2.5 [0.4, 5.1] |
| qwen14b | hotpot | truth | 57 | 3 | 2 | -1.8 [-10.5, 5.3] | +1.8 [0.0, 4.4] |
| qwen14b | hotpot | selection | 124 | 7 | 4 | -2.4 [-8.1, 2.4] | +0.4 [-1.2, 2.0] |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency | 124 | 12 | 5 | -5.7 [-12.1, 0.8] | +0.8 [-1.2, 3.2] |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation | 102 | 7 | 9 | +2.0 [-5.9, 9.8] | +0.0 [-2.5, 2.5] |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 101 | 8 | 8 | +0.0 [-7.9, 7.9] | -1.0 [-4.0, 1.5] |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral | 247 | 13 | 25 | +4.9 [0.0, 9.7] | +2.2 [0.4, 4.0] |
| qwen14b | hotpot | fixed_saturation_vs_neutral | 247 | 15 | 18 | +1.2 [-3.2, 5.7] | +1.6 [-0.4, 3.6] |
| qwen14b | hotpot | bridge_k3_vs_neutral | 124 | 6 | 5 | -0.8 [-6.5, 4.0] | +3.6 [1.2, 6.5] |
| qwen14b | hotpot | bridge_k1_vs_neutral | 124 | 6 | 8 | +1.6 [-4.0, 7.3] | +3.2 [1.2, 5.7] |
| qwen14b | hotpot | entity_vs_neutral | 127 | 8 | 6 | -1.6 [-7.1, 3.9] | +1.2 [-0.8, 3.1] |
| qwen14b | hotpot | unsupported_vs_neutral | 57 | 4 | 3 | -1.8 [-10.5, 7.0] | +1.8 [0.0, 4.4] |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral | 102 | 8 | 8 | +0.0 [-7.8, 7.8] | +2.9 [0.5, 5.4] |

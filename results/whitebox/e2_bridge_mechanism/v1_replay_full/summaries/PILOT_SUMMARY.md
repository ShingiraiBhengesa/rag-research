# Pilot summary (generated)

Runs are never truncated: every run reaches final_report_generation. PCR@30 = end of budget.

## PRIMARY (per cell: premature stop if the baseline gate PASSED, else stopped by step k), paired per question

| victim | dataset | contrast | tier | n | cue rate | control rate | diff [95% CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|
| qwen14b | 2wiki | structure [premature_stop_no_hop2_evidence] | primary | 174 | 0.035 | 0.046 | -1.1 [-4.0, 1.7] | 0.6875 |
| qwen14b | 2wiki | truth [premature_stop_no_hop2_evidence] | primary | 146 | 0.041 | 0.055 | -1.4 [-4.8, 2.1] | 0.6875 |
| qwen14b | 2wiki | selection [premature_stop_no_hop2_evidence] | primary | 191 | 0.042 | 0.063 | -2.1 [-5.2, 0.5] | 0.2891 |
| qwen14b | 2wiki | bridge_k3_vs_fixed_sufficiency [premature_stop_no_hop2_evidence] | secondary | 191 | 0.042 | 0.058 | -1.6 [-4.2, 1.1] | 0.4531 |
| qwen14b | 2wiki | bridge_sat_k3_vs_fixed_saturation [premature_stop_no_hop2_evidence] | secondary | 162 | 0.043 | 0.062 | -1.8 [-4.3, 0.6] | 0.375 |
| qwen14b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 [premature_stop_no_hop2_evidence] | secondary | 162 | 0.043 | 0.043 | +0.0 [-3.1, 3.1] | 1 |
| qwen14b | 2wiki | fixed_sufficiency_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 327 | 0.095 | 0.080 | +1.5 [-0.9, 4.0] | 0.3323 |
| qwen14b | 2wiki | fixed_saturation_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 327 | 0.083 | 0.080 | +0.3 [-2.5, 3.1] | 1 |
| qwen14b | 2wiki | bridge_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 191 | 0.042 | 0.042 | +0.0 [-3.1, 3.1] | 1 |
| qwen14b | 2wiki | bridge_k1_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 191 | 0.063 | 0.042 | +2.1 [-0.5, 5.2] | 0.2891 |
| qwen14b | 2wiki | entity_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 180 | 0.056 | 0.044 | +1.1 [-2.2, 4.4] | 0.7266 |
| qwen14b | 2wiki | unsupported_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 146 | 0.055 | 0.048 | +0.7 [-2.7, 4.1] | 1 |
| qwen14b | 2wiki | bridge_sat_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 162 | 0.043 | 0.049 | -0.6 [-3.1, 1.8] | 1 |
| qwen14b | hotpot | structure [premature_stop_no_hop2_evidence] | primary | 104 | 0.173 | 0.173 | +0.0 [-6.7, 6.7] | 1 |
| qwen14b | hotpot | truth [premature_stop_no_hop2_evidence] | primary | 51 | 0.157 | 0.157 | +0.0 [-9.8, 9.8] | 1 |
| qwen14b | hotpot | selection [premature_stop_no_hop2_evidence] | primary | 109 | 0.174 | 0.156 | +1.8 [-2.8, 7.3] | 0.7266 |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency [premature_stop_no_hop2_evidence] | secondary | 109 | 0.174 | 0.202 | -2.8 [-9.2, 3.7] | 0.5811 |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation [premature_stop_no_hop2_evidence] | secondary | 95 | 0.210 | 0.200 | +1.1 [-5.3, 7.4] | 1 |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 [premature_stop_no_hop2_evidence] | secondary | 94 | 0.213 | 0.181 | +3.2 [-3.2, 9.6] | 0.5078 |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 238 | 0.248 | 0.198 | +5.0 [0.4, 9.7] | 0.04277 |
| qwen14b | hotpot | fixed_saturation_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 238 | 0.261 | 0.198 | +6.3 [1.7, 10.9] | 0.01067 |
| qwen14b | hotpot | bridge_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 109 | 0.174 | 0.156 | +1.8 [-5.5, 8.3] | 0.7905 |
| qwen14b | hotpot | bridge_k1_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 109 | 0.156 | 0.156 | +0.0 [-7.3, 6.4] | 1 |
| qwen14b | hotpot | entity_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 115 | 0.183 | 0.139 | +4.3 [-3.5, 11.3] | 0.3593 |
| qwen14b | hotpot | unsupported_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 51 | 0.157 | 0.196 | -3.9 [-13.7, 5.9] | 0.6875 |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral [premature_stop_no_hop2_evidence] | secondary | 95 | 0.210 | 0.158 | +5.3 [-2.1, 12.6] | 0.2668 |
| llama8b | 2wiki | structure [stopped_by_step_2] | primary | 201 | 0.801 | 0.861 | -6.0 [-12.4, 0.0] | 0.07295 |
| llama8b | 2wiki | truth [stopped_by_step_2] | primary | 192 | 0.812 | 0.844 | -3.1 [-8.3, 2.6] | 0.3449 |
| llama8b | 2wiki | selection [stopped_by_step_2] | primary | 246 | 0.793 | 0.817 | -2.4 [-6.1, 1.6] | 0.3269 |
| llama8b | 2wiki | bridge_k3_vs_fixed_sufficiency [stopped_by_step_2] | secondary | 246 | 0.793 | 0.837 | -4.5 [-10.2, 1.6] | 0.169 |
| llama8b | 2wiki | bridge_sat_k3_vs_fixed_saturation [stopped_by_step_2] | secondary | 211 | 0.716 | 0.915 | -19.9 [-26.1, -13.7] | 0 |
| llama8b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 [stopped_by_step_2] | secondary | 210 | 0.719 | 0.790 | -7.1 [-13.3, -0.5] | 0.04438 |
| llama8b | 2wiki | fixed_sufficiency_vs_neutral [stopped_by_step_2] | secondary | 439 | 0.825 | 0.219 | +60.6 [55.8, 65.4] | 0 |
| llama8b | 2wiki | fixed_saturation_vs_neutral [stopped_by_step_2] | secondary | 439 | 0.909 | 0.219 | +69.0 [64.7, 73.4] | 0 |
| llama8b | 2wiki | bridge_k3_vs_neutral [stopped_by_step_2] | secondary | 242 | 0.789 | 0.256 | +53.3 [45.9, 60.3] | 0 |
| llama8b | 2wiki | bridge_k1_vs_neutral [stopped_by_step_2] | secondary | 242 | 0.814 | 0.256 | +55.8 [48.4, 62.8] | 0 |
| llama8b | 2wiki | entity_vs_neutral [stopped_by_step_2] | secondary | 210 | 0.862 | 0.252 | +61.0 [53.8, 68.6] | 0 |
| llama8b | 2wiki | unsupported_vs_neutral [stopped_by_step_2] | secondary | 190 | 0.842 | 0.242 | +60.0 [52.1, 67.4] | 0 |
| llama8b | 2wiki | bridge_sat_k3_vs_neutral [stopped_by_step_2] | secondary | 209 | 0.713 | 0.278 | +43.5 [34.9, 52.1] | 0 |
| llama8b | hotpot | structure [stopped_by_step_2] | primary | 190 | 0.853 | 0.842 | +1.1 [-4.7, 6.8] | 0.8506 |
| llama8b | hotpot | truth [stopped_by_step_2] | primary | 96 | 0.833 | 0.802 | +3.1 [-4.2, 9.4] | 0.5488 |
| llama8b | hotpot | selection [stopped_by_step_2] | primary | 204 | 0.848 | 0.838 | +1.0 [-2.9, 4.4] | 0.7905 |
| llama8b | hotpot | bridge_k3_vs_fixed_sufficiency [stopped_by_step_2] | secondary | 204 | 0.848 | 0.858 | -1.0 [-6.9, 4.9] | 0.8714 |
| llama8b | hotpot | bridge_sat_k3_vs_fixed_saturation [stopped_by_step_2] | secondary | 182 | 0.824 | 0.923 | -9.9 [-16.0, -3.9] | 0.005098 |
| llama8b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 [stopped_by_step_2] | secondary | 181 | 0.823 | 0.856 | -3.3 [-8.8, 1.7] | 0.3075 |
| llama8b | hotpot | fixed_sufficiency_vs_neutral [stopped_by_step_2] | secondary | 432 | 0.847 | 0.289 | +55.8 [50.7, 60.9] | 0 |
| llama8b | hotpot | fixed_saturation_vs_neutral [stopped_by_step_2] | secondary | 432 | 0.914 | 0.289 | +62.5 [57.6, 67.4] | 0 |
| llama8b | hotpot | bridge_k3_vs_neutral [stopped_by_step_2] | secondary | 202 | 0.847 | 0.346 | +50.0 [42.1, 57.4] | 0 |
| llama8b | hotpot | bridge_k1_vs_neutral [stopped_by_step_2] | secondary | 202 | 0.837 | 0.346 | +49.0 [41.1, 56.9] | 0 |
| llama8b | hotpot | entity_vs_neutral [stopped_by_step_2] | secondary | 205 | 0.839 | 0.346 | +49.3 [42.4, 56.6] | 0 |
| llama8b | hotpot | unsupported_vs_neutral [stopped_by_step_2] | secondary | 96 | 0.802 | 0.354 | +44.8 [33.3, 55.2] | 0 |
| llama8b | hotpot | bridge_sat_k3_vs_neutral [stopped_by_step_2] | secondary | 180 | 0.822 | 0.367 | +45.6 [37.8, 53.3] | 0 |

## Per arm: primary rate and secondary outcomes

| victim | dataset | arm | runs | primary n | premature stop | abstention | search calls after d1 |
|---|---|---|---|---|---|---|---|
| llama8b | 2wiki | bridge_saturation_k3 | 211 | 94 | 0.7447 | 0.0 | 1.578 |
| llama8b | 2wiki | bridge_sufficiency_k1 | 246 | 115 | 0.713 | 0.0 | 1.756 |
| llama8b | 2wiki | bridge_sufficiency_k3 | 246 | 115 | 0.6783 | 0.0 | 1.654 |
| llama8b | 2wiki | entity_sufficiency | 213 | 108 | 0.7685 | 0.0 | 1.62 |
| llama8b | 2wiki | fixed_saturation | 447 | 268 | 0.7388 | 0.0 | 1.584 |
| llama8b | 2wiki | fixed_sufficiency | 447 | 268 | 0.7015 | 0.0 | 1.738 |
| llama8b | 2wiki | neutral | 447 | 265 | 0.5472 | 0.0 | 2.409 |
| llama8b | 2wiki | unsupported_sufficiency | 192 | 91 | 0.7033 | 0.0 | 1.807 |
| llama8b | hotpot | bridge_saturation_k3 | 183 | 70 | 0.7143 | 0.0 | 1.301 |
| llama8b | hotpot | bridge_sufficiency_k1 | 205 | 79 | 0.7468 | 0.0 | 1.224 |
| llama8b | hotpot | bridge_sufficiency_k3 | 205 | 79 | 0.7468 | 0.0 | 1.273 |
| llama8b | hotpot | entity_sufficiency | 209 | 79 | 0.7089 | 0.0 | 1.388 |
| llama8b | hotpot | fixed_saturation | 439 | 216 | 0.7917 | 0.0 | 1.499 |
| llama8b | hotpot | fixed_sufficiency | 439 | 216 | 0.7963 | 0.0 | 1.437 |
| llama8b | hotpot | neutral | 439 | 215 | 0.5953 | 0.0 | 2.23 |
| llama8b | hotpot | unsupported_sufficiency | 96 | 42 | 0.8333 | 0.0 | 1.458 |
| qwen14b | 2wiki | bridge_saturation_k3 | 230 | 162 | 0.0432 | 0.0043 | 4.57 |
| qwen14b | 2wiki | bridge_sufficiency_k1 | 268 | 191 | 0.0628 | 0.0 | 4.701 |
| qwen14b | 2wiki | bridge_sufficiency_k3 | 268 | 191 | 0.0419 | 0.0075 | 4.862 |
| qwen14b | 2wiki | entity_sufficiency | 246 | 180 | 0.0556 | 0.0081 | 4.443 |
| qwen14b | 2wiki | fixed_saturation | 450 | 327 | 0.0826 | 0.0022 | 5.176 |
| qwen14b | 2wiki | fixed_sufficiency | 450 | 327 | 0.0948 | 0.0067 | 4.938 |
| qwen14b | 2wiki | neutral | 450 | 327 | 0.0795 | 0.0022 | 5.309 |
| qwen14b | 2wiki | unsupported_sufficiency | 201 | 146 | 0.0548 | 0.0 | 5.239 |
| qwen14b | hotpot | bridge_saturation_k3 | 203 | 95 | 0.2105 | 0.0099 | 4.798 |
| qwen14b | hotpot | bridge_sufficiency_k1 | 235 | 109 | 0.156 | 0.0043 | 5.17 |
| qwen14b | hotpot | bridge_sufficiency_k3 | 235 | 109 | 0.1743 | 0.0043 | 5.574 |
| qwen14b | hotpot | entity_sufficiency | 240 | 115 | 0.1826 | 0.0042 | 6.471 |
| qwen14b | hotpot | fixed_saturation | 450 | 238 | 0.2605 | 0.0067 | 5.736 |
| qwen14b | hotpot | fixed_sufficiency | 450 | 238 | 0.2479 | 0.0022 | 6.0 |
| qwen14b | hotpot | neutral | 450 | 239 | 0.1967 | 0.0067 | 6.173 |
| qwen14b | hotpot | unsupported_sufficiency | 106 | 51 | 0.1569 | 0.0094 | 4.547 |

## PCR contrasts (all paired questions; Δ in pp, 95% paired bootstrap CI)

| victim | dataset | contrast | n | Δ@1 | Δ@2 | Δ@5 | Δ@30 |
|---|---|---|---|---|---|---|---|
| llama8b | 2wiki | bridge_k1_vs_neutral | 242 | -0.8 [-2.1, 0.0] | +55.8 [48.4, 62.8] | +18.6 [13.2, 24.0] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_k3_vs_fixed_sufficiency | 246 | -0.8 [-2.0, 0.0] | -4.5 [-10.2, 1.6] | +0.0 [-2.9, 2.9] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_k3_vs_neutral | 242 | -0.8 [-2.1, 0.0] | +53.3 [45.9, 60.3] | +19.4 [14.1, 24.8] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 210 | +0.0 [0.0, 0.0] | -7.1 [-13.3, -0.5] | -2.4 [-5.7, 0.5] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 211 | -1.4 [-3.3, 0.0] | -19.9 [-26.1, -13.7] | -4.3 [-7.6, -0.9] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_neutral | 209 | -1.0 [-2.4, 0.0] | +43.5 [34.9, 52.1] | +15.8 [10.5, 21.5] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | entity_vs_neutral | 210 | -0.9 [-2.4, 0.0] | +61.0 [53.8, 68.6] | +17.1 [11.9, 22.9] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | fixed_saturation_vs_neutral | 439 | -0.5 [-1.6, 0.7] | +69.0 [64.7, 73.4] | +27.8 [23.5, 32.1] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | fixed_sufficiency_vs_neutral | 439 | -0.7 [-1.8, 0.5] | +60.6 [55.8, 65.4] | +24.4 [19.8, 28.9] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | selection | 246 | +0.0 [0.0, 0.0] | -2.4 [-6.1, 1.6] | +0.8 [-0.8, 2.9] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | structure | 201 | +0.0 [0.0, 0.0] | -6.0 [-12.4, 0.0] | +1.0 [-2.0, 4.0] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | truth | 192 | +0.0 [0.0, 0.0] | -3.1 [-8.3, 2.6] | +3.1 [0.0, 6.2] | +0.0 [0.0, 0.0] |
| llama8b | 2wiki | unsupported_vs_neutral | 190 | -1.1 [-2.6, 0.0] | +60.0 [52.1, 67.4] | +17.9 [12.1, 24.2] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_k1_vs_neutral | 202 | -4.0 [-6.9, -1.0] | +49.0 [41.1, 56.9] | +14.8 [10.4, 19.8] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_k3_vs_fixed_sufficiency | 204 | -1.5 [-3.4, 0.0] | -1.0 [-6.9, 4.9] | -2.0 [-4.4, 0.5] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_k3_vs_neutral | 202 | -4.5 [-7.4, -2.0] | +50.0 [42.1, 57.4] | +14.4 [9.9, 19.3] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 181 | +1.1 [0.0, 2.8] | -3.3 [-8.8, 1.7] | -0.5 [-3.9, 2.8] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_sat_k3_vs_fixed_saturation | 182 | -1.1 [-2.8, 0.0] | -9.9 [-16.0, -3.9] | -3.3 [-6.6, -0.5] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | bridge_sat_k3_vs_neutral | 180 | -3.9 [-7.8, -0.6] | +45.6 [37.8, 53.3] | +13.3 [8.3, 18.9] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | entity_vs_neutral | 205 | -4.4 [-7.8, -1.5] | +49.3 [42.4, 56.6] | +15.1 [10.2, 20.5] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | fixed_saturation_vs_neutral | 432 | -2.3 [-4.2, -0.5] | +62.5 [57.6, 67.4] | +20.4 [16.7, 24.3] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | fixed_sufficiency_vs_neutral | 432 | -2.3 [-4.2, -0.7] | +55.8 [50.7, 60.9] | +17.1 [13.4, 21.1] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | selection | 204 | -0.5 [-1.5, 0.0] | +1.0 [-2.9, 4.4] | -0.5 [-1.5, 0.0] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | structure | 190 | +0.0 [-1.6, 1.6] | +1.1 [-4.7, 6.8] | -1.1 [-4.2, 2.1] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | truth | 96 | -1.0 [-3.1, 0.0] | +3.1 [-4.2, 9.4] | +6.2 [2.1, 11.5] | +0.0 [0.0, 0.0] |
| llama8b | hotpot | unsupported_vs_neutral | 96 | -3.1 [-8.3, 1.0] | +44.8 [33.3, 55.2] | +7.3 [-1.0, 14.6] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_k1_vs_neutral | 268 | +18.7 [13.1, 24.6] | +25.0 [19.4, 30.6] | +1.9 [-1.1, 4.9] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_k3_vs_fixed_sufficiency | 267 | -8.2 [-13.9, -2.2] | -6.7 [-10.9, -2.6] | -1.9 [-4.5, 0.4] | +0.4 [0.0, 1.1] |
| qwen14b | 2wiki | bridge_k3_vs_neutral | 268 | +26.1 [20.5, 32.1] | +26.1 [20.5, 32.1] | +2.2 [-0.8, 5.6] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 228 | -0.4 [-6.6, 6.1] | +4.8 [0.4, 9.2] | +0.9 [-2.2, 4.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 228 | -2.2 [-8.3, 3.5] | -0.4 [-4.8, 4.0] | -0.9 [-4.4, 2.6] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | bridge_sat_k3_vs_neutral | 229 | +27.5 [20.9, 34.1] | +29.7 [23.6, 35.8] | +3.1 [-0.4, 7.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | entity_vs_neutral | 244 | +10.2 [4.5, 16.0] | +16.8 [11.1, 22.5] | -1.2 [-4.5, 2.1] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | fixed_saturation_vs_neutral | 449 | +24.7 [20.5, 29.2] | +27.8 [23.4, 32.3] | +5.8 [3.1, 8.5] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | fixed_sufficiency_vs_neutral | 449 | +30.3 [25.6, 35.0] | +30.7 [26.5, 35.2] | +6.9 [4.0, 9.8] | -0.2 [-0.7, 0.0] |
| qwen14b | 2wiki | selection | 268 | +7.5 [3.0, 12.3] | +1.1 [-1.5, 4.1] | +0.4 [-1.9, 2.6] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | structure | 235 | +16.6 [10.6, 22.6] | +7.7 [2.1, 13.2] | +3.4 [0.9, 6.0] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | truth | 201 | +12.9 [6.5, 18.9] | +5.0 [-1.0, 10.9] | -1.5 [-5.0, 1.5] | +0.0 [0.0, 0.0] |
| qwen14b | 2wiki | unsupported_vs_neutral | 201 | +14.4 [8.5, 20.9] | +20.9 [14.4, 27.9] | +3.5 [0.0, 7.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k1_vs_neutral | 235 | +18.3 [12.8, 23.8] | +27.7 [21.7, 33.6] | +14.0 [8.9, 19.6] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency | 235 | -11.1 [-16.6, -5.5] | -7.7 [-14.0, -1.3] | -5.5 [-10.6, -0.4] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_neutral | 235 | +18.3 [13.2, 23.8] | +23.4 [17.4, 29.8] | +11.5 [6.0, 17.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 200 | -2.0 [-8.5, 4.5] | +8.5 [3.5, 14.0] | +6.0 [1.0, 10.5] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation | 202 | -6.4 [-12.9, 0.5] | +3.0 [-3.5, 9.4] | +0.5 [-4.0, 5.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral | 202 | +17.8 [11.9, 24.3] | +32.2 [26.2, 39.1] | +17.8 [11.9, 23.8] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | entity_vs_neutral | 239 | +7.5 [2.9, 12.1] | +17.2 [11.7, 22.6] | +10.9 [5.9, 15.9] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_saturation_vs_neutral | 449 | +24.7 [20.5, 28.7] | +30.1 [25.6, 34.5] | +15.8 [11.8, 20.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral | 449 | +29.8 [25.4, 33.9] | +31.4 [27.2, 35.9] | +18.3 [14.2, 22.3] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | selection | 235 | +0.0 [-4.3, 4.3] | -4.3 [-8.5, -0.4] | -2.5 [-6.4, 1.3] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | structure | 222 | +10.4 [4.5, 16.2] | +5.4 [0.4, 10.4] | +1.4 [-3.6, 5.9] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | truth | 106 | +1.9 [-5.7, 8.5] | +6.6 [-0.9, 15.1] | -0.9 [-8.5, 6.6] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | unsupported_vs_neutral | 106 | +13.2 [5.7, 21.7] | +11.3 [3.8, 18.9] | +3.8 [-2.8, 11.3] | +0.0 [0.0, 0.0] |

## Final-answer accuracy and evidence coverage by arm

| victim | dataset | arm | n | accuracy | mean evidence coverage | full coverage |
|---|---|---|---|---|---|---|
| llama8b | 2wiki | bridge_saturation_k3 | 211 | 0.3223 | 0.8555 | 0.6635 |
| llama8b | 2wiki | bridge_sufficiency_k1 | 246 | 0.3171 | 0.8526 | 0.6504 |
| llama8b | 2wiki | bridge_sufficiency_k3 | 246 | 0.3008 | 0.8557 | 0.6545 |
| llama8b | 2wiki | entity_sufficiency | 213 | 0.3052 | 0.8275 | 0.5962 |
| llama8b | 2wiki | fixed_saturation | 447 | 0.2774 | 0.7377 | 0.4899 |
| llama8b | 2wiki | fixed_sufficiency | 447 | 0.255 | 0.755 | 0.5324 |
| llama8b | 2wiki | neutral | 447 | 0.2729 | 0.7981 | 0.5951 |
| llama8b | 2wiki | unsupported_sufficiency | 192 | 0.2865 | 0.8477 | 0.6354 |
| llama8b | hotpot | bridge_saturation_k3 | 183 | 0.3169 | 0.8607 | 0.7268 |
| llama8b | hotpot | bridge_sufficiency_k1 | 205 | 0.3268 | 0.8512 | 0.7073 |
| llama8b | hotpot | bridge_sufficiency_k3 | 205 | 0.3171 | 0.8512 | 0.7073 |
| llama8b | hotpot | entity_sufficiency | 209 | 0.2919 | 0.8589 | 0.7273 |
| llama8b | hotpot | fixed_saturation | 439 | 0.2711 | 0.7677 | 0.5923 |
| llama8b | hotpot | fixed_sufficiency | 439 | 0.2938 | 0.7711 | 0.6014 |
| llama8b | hotpot | neutral | 439 | 0.3075 | 0.8246 | 0.6834 |
| llama8b | hotpot | unsupported_sufficiency | 96 | 0.25 | 0.8177 | 0.6354 |
| qwen14b | 2wiki | bridge_saturation_k3 | 230 | 0.4565 | 0.9793 | 0.9478 |
| qwen14b | 2wiki | bridge_sufficiency_k1 | 268 | 0.4478 | 0.972 | 0.9328 |
| qwen14b | 2wiki | bridge_sufficiency_k3 | 268 | 0.459 | 0.9804 | 0.9515 |
| qwen14b | 2wiki | entity_sufficiency | 246 | 0.439 | 0.9726 | 0.935 |
| qwen14b | 2wiki | fixed_saturation | 450 | 0.4222 | 0.9622 | 0.9133 |
| qwen14b | 2wiki | fixed_sufficiency | 450 | 0.4378 | 0.9578 | 0.9044 |
| qwen14b | 2wiki | neutral | 450 | 0.4444 | 0.9622 | 0.9133 |
| qwen14b | 2wiki | unsupported_sufficiency | 201 | 0.4229 | 0.9751 | 0.9453 |
| qwen14b | hotpot | bridge_saturation_k3 | 203 | 0.4483 | 0.9483 | 0.8966 |
| qwen14b | hotpot | bridge_sufficiency_k1 | 235 | 0.4128 | 0.9596 | 0.9191 |
| qwen14b | hotpot | bridge_sufficiency_k3 | 235 | 0.4468 | 0.9553 | 0.9106 |
| qwen14b | hotpot | entity_sufficiency | 240 | 0.3708 | 0.9542 | 0.9083 |
| qwen14b | hotpot | fixed_saturation | 450 | 0.3711 | 0.9267 | 0.8556 |
| qwen14b | hotpot | fixed_sufficiency | 450 | 0.3822 | 0.9289 | 0.86 |
| qwen14b | hotpot | neutral | 450 | 0.36 | 0.9433 | 0.8889 |
| qwen14b | hotpot | unsupported_sufficiency | 106 | 0.4528 | 0.9623 | 0.9245 |

## Paired answer flips and evidence loss per contrast

| victim | dataset | contrast | n | ctl✓→cue✗ | ctl✗→cue✓ | Δacc [CI] | evidence loss [CI] |
|---|---|---|---|---|---|---|---|
| llama8b | 2wiki | structure | 201 | 14 | 13 | -0.5 [-6.0, 4.5] | -2.1 [-4.3, -0.1] |
| llama8b | 2wiki | truth | 192 | 18 | 15 | -1.6 [-7.3, 4.2] | -0.1 [-1.8, 1.6] |
| llama8b | 2wiki | selection | 246 | 13 | 9 | -1.6 [-5.3, 2.0] | -0.3 [-1.4, 0.8] |
| llama8b | 2wiki | bridge_k3_vs_fixed_sufficiency | 246 | 17 | 21 | +1.6 [-3.2, 6.5] | -0.7 [-2.5, 1.1] |
| llama8b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 211 | 12 | 13 | +0.5 [-4.3, 5.2] | -1.2 [-3.1, 0.7] |
| llama8b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 210 | 10 | 17 | +3.3 [-1.4, 8.1] | +1.1 [-0.5, 2.9] |
| llama8b | 2wiki | fixed_sufficiency_vs_neutral | 447 | 36 | 28 | -1.8 [-5.4, 1.6] | +4.3 [2.7, 6.0] |
| llama8b | 2wiki | fixed_saturation_vs_neutral | 447 | 30 | 32 | +0.4 [-3.1, 4.0] | +6.0 [4.4, 7.8] |
| llama8b | 2wiki | bridge_k3_vs_neutral | 246 | 17 | 18 | +0.4 [-4.1, 5.3] | +1.4 [-0.7, 3.4] |
| llama8b | 2wiki | bridge_k1_vs_neutral | 246 | 13 | 18 | +2.0 [-2.4, 6.5] | +1.7 [-0.2, 3.7] |
| llama8b | 2wiki | entity_vs_neutral | 213 | 16 | 16 | +0.0 [-5.2, 5.2] | +3.2 [1.2, 5.3] |
| llama8b | 2wiki | unsupported_vs_neutral | 192 | 15 | 17 | +1.0 [-4.7, 6.2] | +1.9 [-0.3, 4.0] |
| llama8b | 2wiki | bridge_sat_k3_vs_neutral | 211 | 11 | 16 | +2.4 [-2.8, 7.1] | +1.8 [-0.6, 4.0] |
| llama8b | hotpot | structure | 191 | 11 | 12 | +0.5 [-4.2, 5.2] | +0.8 [-1.1, 2.6] |
| llama8b | hotpot | truth | 96 | 4 | 8 | +4.2 [-3.1, 11.5] | -1.0 [-3.6, 1.6] |
| llama8b | hotpot | selection | 205 | 7 | 5 | -1.0 [-4.4, 1.9] | +0.0 [-1.7, 1.7] |
| llama8b | hotpot | bridge_k3_vs_fixed_sufficiency | 205 | 12 | 14 | +1.0 [-3.9, 5.9] | -1.2 [-3.4, 0.7] |
| llama8b | hotpot | bridge_sat_k3_vs_fixed_saturation | 183 | 13 | 20 | +3.8 [-2.7, 10.4] | -1.6 [-3.8, 0.5] |
| llama8b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 182 | 9 | 12 | +1.7 [-3.3, 6.6] | -1.1 [-3.3, 0.8] |
| llama8b | hotpot | fixed_sufficiency_vs_neutral | 439 | 36 | 30 | -1.4 [-5.0, 2.3] | +5.3 [3.3, 7.4] |
| llama8b | hotpot | fixed_saturation_vs_neutral | 439 | 37 | 21 | -3.6 [-7.1, -0.2] | +5.7 [3.9, 7.7] |
| llama8b | hotpot | bridge_k3_vs_neutral | 205 | 13 | 12 | -0.5 [-5.4, 4.4] | +2.7 [0.5, 5.1] |
| llama8b | hotpot | bridge_k1_vs_neutral | 205 | 16 | 17 | +0.5 [-4.9, 5.9] | +2.7 [0.2, 5.1] |
| llama8b | hotpot | entity_vs_neutral | 209 | 18 | 15 | -1.4 [-6.7, 4.3] | +1.2 [-1.2, 3.6] |
| llama8b | hotpot | unsupported_vs_neutral | 96 | 13 | 7 | -6.2 [-15.6, 3.1] | +3.6 [0.5, 6.8] |
| llama8b | hotpot | bridge_sat_k3_vs_neutral | 183 | 9 | 11 | +1.1 [-3.8, 6.0] | +1.6 [-0.8, 4.1] |
| qwen14b | 2wiki | structure | 236 | 7 | 8 | +0.4 [-3.0, 3.8] | -0.6 [-1.8, 0.4] |
| qwen14b | 2wiki | truth | 201 | 12 | 14 | +1.0 [-4.0, 6.5] | -0.5 [-1.9, 0.9] |
| qwen14b | 2wiki | selection | 268 | 7 | 10 | +1.1 [-1.9, 4.1] | -0.8 [-2.0, 0.2] |
| qwen14b | 2wiki | bridge_k3_vs_fixed_sufficiency | 268 | 16 | 14 | -0.8 [-4.9, 3.7] | -0.8 [-1.8, 0.3] |
| qwen14b | 2wiki | bridge_sat_k3_vs_fixed_saturation | 230 | 7 | 12 | +2.2 [-1.3, 5.7] | -0.7 [-1.7, 0.2] |
| qwen14b | 2wiki | bridge_sat_k3_vs_bridge_suff_k3 | 229 | 9 | 10 | +0.4 [-3.1, 3.9] | +0.0 [-1.3, 1.1] |
| qwen14b | 2wiki | fixed_sufficiency_vs_neutral | 450 | 32 | 29 | -0.7 [-4.0, 2.9] | +0.4 [-0.4, 1.4] |
| qwen14b | 2wiki | fixed_saturation_vs_neutral | 450 | 34 | 24 | -2.2 [-5.6, 1.1] | +0.0 [-0.9, 1.1] |
| qwen14b | 2wiki | bridge_k3_vs_neutral | 268 | 14 | 9 | -1.9 [-5.2, 1.9] | -0.4 [-1.7, 0.8] |
| qwen14b | 2wiki | bridge_k1_vs_neutral | 268 | 19 | 11 | -3.0 [-7.1, 1.1] | +0.5 [-0.6, 1.5] |
| qwen14b | 2wiki | entity_vs_neutral | 246 | 13 | 6 | -2.9 [-6.1, 0.8] | +0.2 [-1.0, 1.5] |
| qwen14b | 2wiki | unsupported_vs_neutral | 201 | 15 | 6 | -4.5 [-9.0, -0.5] | -0.2 [-1.6, 1.1] |
| qwen14b | 2wiki | bridge_sat_k3_vs_neutral | 230 | 14 | 8 | -2.6 [-6.5, 1.3] | -0.7 [-2.0, 0.4] |
| qwen14b | hotpot | structure | 223 | 10 | 21 | +4.9 [0.0, 9.4] | +0.2 [-1.4, 1.8] |
| qwen14b | hotpot | truth | 106 | 4 | 4 | +0.0 [-5.7, 5.7] | +0.0 [-2.4, 2.4] |
| qwen14b | hotpot | selection | 235 | 9 | 17 | +3.4 [-0.4, 7.7] | +0.4 [-0.9, 1.7] |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency | 235 | 11 | 19 | +3.4 [-0.9, 7.7] | -0.6 [-2.1, 0.9] |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation | 203 | 5 | 11 | +3.0 [-1.0, 6.9] | +0.2 [-1.2, 1.7] |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 | 201 | 12 | 7 | -2.5 [-6.5, 1.5] | +0.8 [-0.8, 2.2] |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral | 450 | 24 | 34 | +2.2 [-1.3, 5.6] | +1.4 [0.2, 2.6] |
| qwen14b | hotpot | fixed_saturation_vs_neutral | 450 | 30 | 35 | +1.1 [-2.4, 4.4] | +1.7 [0.4, 2.8] |
| qwen14b | hotpot | bridge_k3_vs_neutral | 235 | 8 | 25 | +7.2 [2.5, 11.9] | +0.4 [-1.3, 1.9] |
| qwen14b | hotpot | bridge_k1_vs_neutral | 235 | 7 | 16 | +3.8 [0.0, 7.7] | +0.0 [-1.5, 1.5] |
| qwen14b | hotpot | entity_vs_neutral | 240 | 13 | 16 | +1.2 [-3.3, 5.8] | +0.8 [-1.0, 2.5] |
| qwen14b | hotpot | unsupported_vs_neutral | 106 | 2 | 8 | +5.7 [0.0, 11.3] | -0.9 [-3.3, 1.4] |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral | 203 | 4 | 14 | +4.9 [1.0, 8.9] | +1.2 [-0.5, 3.0] |

# Pilot summary (generated)

Runs are never truncated: every run reaches final_report_generation. PCR@30 = end of budget.

## PRIMARY (per cell: premature stop if the baseline gate PASSED, else stopped by step k), paired per question

| victim | dataset | contrast | tier | n | cue rate | control rate | diff [95% CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|
| qwen14b | hotpot | structure [stopped_by_step_2] | primary | 1 | 0.000 | 0.000 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | truth [stopped_by_step_2] | primary | 0 | | | | |
| qwen14b | hotpot | selection [stopped_by_step_2] | primary | 1 | 0.000 | 0.000 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency [stopped_by_step_2] | secondary | 1 | 0.000 | 0.000 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | bridge_sat_k3_vs_fixed_saturation [stopped_by_step_2] | secondary | 0 | | | | |
| qwen14b | hotpot | bridge_sat_k3_vs_bridge_suff_k3 [stopped_by_step_2] | secondary | 0 | | | | |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral [stopped_by_step_2] | secondary | 3 | 0.667 | 0.667 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | fixed_saturation_vs_neutral [stopped_by_step_2] | secondary | 3 | 0.667 | 0.667 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | bridge_k3_vs_neutral [stopped_by_step_2] | secondary | 1 | 0.000 | 0.000 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | bridge_k1_vs_neutral [stopped_by_step_2] | secondary | 1 | 0.000 | 0.000 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | entity_vs_neutral [stopped_by_step_2] | secondary | 1 | 0.000 | 0.000 | +0.0 [0.0, 0.0] | 1 |
| qwen14b | hotpot | unsupported_vs_neutral [stopped_by_step_2] | secondary | 0 | | | | |
| qwen14b | hotpot | bridge_sat_k3_vs_neutral [stopped_by_step_2] | secondary | 0 | | | | |

## Per arm: primary rate and secondary outcomes

| victim | dataset | arm | runs | primary n | premature stop | abstention | search calls after d1 |
|---|---|---|---|---|---|---|---|
| qwen14b | hotpot | bridge_saturation_k3 | 0 | 0 | None | None | None |
| qwen14b | hotpot | bridge_sufficiency_k1 | 1 | 0 | None | 1.0 | 2.0 |
| qwen14b | hotpot | bridge_sufficiency_k3 | 1 | 0 | None | 0.0 | 10.0 |
| qwen14b | hotpot | entity_sufficiency | 1 | 0 | None | 0.0 | 9.0 |
| qwen14b | hotpot | fixed_saturation | 3 | 0 | None | 0.3333 | 5.333 |
| qwen14b | hotpot | fixed_sufficiency | 3 | 0 | None | 0.0 | 5.0 |
| qwen14b | hotpot | neutral | 3 | 0 | None | 0.0 | 6.333 |
| qwen14b | hotpot | unsupported_sufficiency | 0 | 0 | None | None | None |

## PCR contrasts (all paired questions; Δ in pp, 95% paired bootstrap CI)

| victim | dataset | contrast | n | Δ@1 | Δ@2 | Δ@5 | Δ@30 |
|---|---|---|---|---|---|---|---|
| qwen14b | hotpot | bridge_k1_vs_neutral | 1 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | -100.0 [-100.0, -100.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency | 1 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +100.0 [100.0, 100.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_neutral | 1 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | entity_vs_neutral | 1 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | -100.0 [-100.0, -100.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_saturation_vs_neutral | 3 | -33.3 [-100.0, 0.0] | +0.0 [0.0, 0.0] | -33.3 [-100.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral | 3 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | -33.3 [-100.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | selection | 1 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +100.0 [100.0, 100.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | structure | 1 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +100.0 [100.0, 100.0] | +0.0 [0.0, 0.0] |

## Final-answer accuracy and evidence coverage by arm

| victim | dataset | arm | n | accuracy | mean evidence coverage | full coverage |
|---|---|---|---|---|---|---|
| qwen14b | hotpot | bridge_sufficiency_k1 | 1 | 0.0 | 1.0 | 1.0 |
| qwen14b | hotpot | bridge_sufficiency_k3 | 1 | 0.0 | 1.0 | 1.0 |
| qwen14b | hotpot | entity_sufficiency | 1 | 0.0 | 1.0 | 1.0 |
| qwen14b | hotpot | fixed_saturation | 3 | 0.3333 | 1.0 | 1.0 |
| qwen14b | hotpot | fixed_sufficiency | 3 | 0.3333 | 1.0 | 1.0 |
| qwen14b | hotpot | neutral | 3 | 0.3333 | 1.0 | 1.0 |

## Paired answer flips and evidence loss per contrast

| victim | dataset | contrast | n | ctl✓→cue✗ | ctl✗→cue✓ | Δacc [CI] | evidence loss [CI] |
|---|---|---|---|---|---|---|---|
| qwen14b | hotpot | structure | 1 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | selection | 1 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_fixed_sufficiency | 1 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_sufficiency_vs_neutral | 3 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | fixed_saturation_vs_neutral | 3 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k3_vs_neutral | 1 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | bridge_k1_vs_neutral | 1 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| qwen14b | hotpot | entity_vs_neutral | 1 | 0 | 0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |

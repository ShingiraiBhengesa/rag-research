# Answer harm — results (v2)

All four E1 models (Qwen2.5-14B, Llama-3.1-8B, Gemma-3-27B, gpt-oss-20b) ×
four datasets: 84,508 runs, each with an answer and an evidence record.

Method and pre-specification: [`SPEC.md`](SPEC.md). Code:
`src/prematurestop/analysis/fixed_cue_matrix/answer_harm.py`. Every number below
names the CSV it comes from.

**Terms.**
- Net harm = (control right & cue wrong − control wrong & cue right) / pairs,
  in percentage points. Positive means the cue made answers worse.
- Accuracy is the automatic semantic label, before human adjudication.

## 1. On average, the cues do not detectably harm final answers

| | Qwen-14B | Llama-8B | Gemma-27B | gpt-oss-20b | All models |
|---|---|---|---|---|---|
| Fixed Sufficiency vs Neutral | +0.7 [−1.2, +2.7] | −0.1 [−2.0, +1.9] | +1.1 [−1.1, +3.4] | −0.1 [−2.2, +2.0] | **+0.4 [−0.6, +1.5]** |
| Sufficiency + Saturation vs Neutral | +0.7 [−1.2, +2.6] | +0.7 [−1.2, +2.6] | −0.2 [−2.4, +2.2] | −1.2 [−3.3, +0.9] | **+0.0 [−1.0, +1.1]** |

Net harm in pp, 95% stratified paired bootstrap CI [table: pooled.csv]. Figure:
`figures/net_harm_by_model.png`.

- No comparison in any family, attack or control, has a pooled CI that excludes
  zero [table: pooled.csv].
- Of 224 model × dataset × comparison cells, 3 have raw McNemar p < 0.05,
  fewer than the about 11 expected by chance, and in both directions. None
  survive Benjamini–Hochberg correction within its family
  [table: cells.csv, column `bh_q`].
- The result does not change when pairs with an adjudication-flagged answer
  are dropped (S1), or when the three gpt-oss datasets served without the
  lenient parser are dropped (S2) [table: sensitivity.csv].
- There is no harm in any hop stratum. There is none whether or not d₁ already
  held all the evidence [table: strata.csv].

## 2. But answers follow evidence: harm is real exactly where the cue costs evidence

Each pair is classified by whether the control and cue runs had full evidence
at the **final-report input**.

| Paired evidence transition | Pairs (Saturation) | Fixed Sufficiency | Sufficiency + Saturation |
|---|---|---|---|
| Both complete | 4,174 | +0.2 | −0.4 |
| **Evidence lost by cue** | 683 | **+9.4** (p < 0.001) | **+9.8** (p < 0.001) |
| **Evidence gained by cue** | 516 | **−10.8** (p < 0.001) | **−12.6** (p < 0.001) |
| Both incomplete | 1,383 | +1.1 | +1.1 |

Net harm in pp, all models pooled, stratified by model × dataset
[table: mechanism.csv, `boundary = final_report_input`, `model = all`]. Figure:
`figures/net_harm_by_evidence_transition.png`.

- The pattern holds in every model.
  - Evidence lost: Qwen +9.0, Llama +9.2, Gemma +16.1 and gpt-oss +5.8 pp
    (Saturation).
  - Evidence gained: −10.7 to −17.3 pp (Saturation; Fixed Sufficiency −5.1 to −14.1).
  - Source: [table: mechanism.csv].
- Within a question, losing evidence costs about 10 points of accuracy and
  gaining it is worth about the same.
- At the Researcher's own stop (`local_stop`) the pattern is the same but
  weaker: +5.4 / +6.8 pp for lost evidence [table: mechanism.csv]. Supervisor
  recovery and compression sit between the two boundaries.

## 3. The sufficiency cues cause a small but significant net evidence loss

| Comparison | Evidence lost | Evidence gained | Net lost (% of pairs) | Sign test |
|---|---|---|---|---|
| Neutral vs Clean (noise baseline) | 432 | 400 | — | p = 0.28 |
| Epistemic Confirmation vs Neutral | 410 | 392 | — | p = 0.55 |
| Search Match vs Neutral | 468 | 441 | — | p = 0.39 |
| **Fixed Sufficiency vs Neutral** | 680 | 539 | **2.1%** | **p = 6×10⁻⁵** |
| **Sufficiency + Saturation vs Neutral** | 683 | 516 | **2.5%** | **p = 2×10⁻⁶** |

All models pooled, final-report input [table: evidence_shift.csv].

- Every model shows a net loss under both primary cues (1.3–3.5% of pairs). It
  is significant individually in 5 of 8 model × cue cells
  [table: evidence_shift.csv].
- The weaker cues do not move evidence beyond the run-to-run noise seen
  between Neutral and Clean.

## 4. How the three results fit together

The pathway is cue → stop earlier → slightly more evidence lost than gained →
each lost question about 10 pp more likely wrong.

- The net loss of about 2.5% of questions × about 10 pp predicts an average
  harm of roughly **+0.25 pp**.
- The pooled CI is about ±1 pp, so the average effect is real in mechanism but
  **below what this study can detect**. With about 19% discordant pairs, 80%
  power at +0.25 pp would need about 240,000 pairs per comparison, about 35×
  the current 6,800 [table: pooled.csv].
- The correct summary is not "the cue causes no harm". It is: the cue causes
  harm in proportion to the evidence it costs. On this system that cost is
  small, because the Supervisor and natural run-to-run variation absorb most
  early stops.
- This also explains why E1's large PCR effects (+9 to +26 pp earlier stopping)
  do not show up as accuracy loss.

## 5. Caveats

- The evidence transition is itself affected by the cue, so section 2 is a
  mediation-style decomposition, not a randomized contrast. Pairing the two
  runs of the same question removes question difficulty as an explanation,
  but other run-level factors could still co-vary with both evidence and
  answer.
- Accuracy is pre-adjudication (`needs_adjudication` counts as not correct).
  Dropping flagged pairs changes nothing (S1).
- Qwen/Llama answers come from
  `local_data/evaluation/v3_final_completed_qwen_llama_20260910/answers/final_answer_predictions.jsonl`.
  It reproduces the published `qwen_llama/data/answer_summary.csv` correct
  counts in 96/96 cells; the bundle's own per-run file is stored in Git LFS.
- gpt-oss: HotpotQA, MuSiQue and MultiHop-RAG were served without the lenient
  Harmony parser, unlike 2Wiki. Dropping them changes nothing (S2).

## Files

| File | Contents |
|---|---|
| `cells.csv` | Net harm per model × dataset × comparison, McNemar p, BH q |
| `pooled.csv` | Net harm pooled per model and over all models |
| `mechanism.csv` | Net harm per evidence-transition class, final and local boundaries |
| `evidence_shift.csv` | Evidence lost vs gained per comparison, sign test (post-hoc) |
| `strata.csv` | d₁-already-complete and hop-count strata |
| `sensitivity.csv` | S1 (adjudication) and S2 (gpt-oss parser) |
| `figures/` | `net_harm_by_model`, `net_harm_by_evidence_transition` (PNG + PDF) |

Regenerate everything (about one minute, no traces read):

    python -m prematurestop.analysis.fixed_cue_matrix.answer_harm

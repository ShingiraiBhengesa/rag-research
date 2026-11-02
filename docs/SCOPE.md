# Scope

This repository accompanies the MSc thesis *Sufficient by Design:
Observation-Level Sufficiency Cues Manipulate Completion Decisions in Agentic
RAG* (Saint Louis University, 2026).

## What the thesis evaluates

Outcomes at the Researcher's **local termination**: when it ends its episode,
and what evidence it holds at that point.

* **Completion timing:** ASR, step reduction D_q and PCR@k.
* **Local premature completion:** EICR, the share of completing runs that stop
  while a required gold evidence unit is still missing.
* **Local evidence loss:** the paired difference in gold-evidence coverage at
  the Researcher's stop.

E1 (fixed-cue matrix) and E2 (bridge-anchored study) use the same local
measures. E2's whole-run **premature stop** (Equation A.1) was pre-specified
as its primary outcome; the thesis reports it in **Appendix A**.

## What it does not evaluate

* **Final-answer accuracy.** Every run is carried through to a final report,
  and the scoring code (`evaluation/answer_scoring.py`) is kept for
  reproducibility. The resulting answer-harm analysis is archived in
  `results/whitebox/_out_of_scope/answer_harm/` and is not a thesis result.
* **Downstream orchestration:** whether an evidence deficit survives
  Supervisor re-delegation or report writing. This is future work (Chapter 6).
* **Black-box transfer** to hosted models. This is future work; no code or
  results for it are in this repository.

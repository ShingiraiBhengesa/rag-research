# Lab meeting brief

Six-slide story (spec section 9). Slide tables here are capped at 4-6 rows;
full condition tables are in the workbook and appendix -- point there, do
not paste them here.

## Slide 1 -- Attack and evaluation question

Does a natural-language cue embedded in a retrieved document change how
quickly and how completely a ReAct-style researcher agent terminates its
search, and if so, does that process-level change translate into lost
evidence or a worse final answer?

## Slide 2 -- Data freeze and scope

- Models: Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct.
- Datasets: HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHopRAG.
- 12 conditions x 4 datasets x 2 models, 42547 valid completed runs
  total [table: data/run_inventory.csv].
- Validation gates: 12/13 passing as of this brief's
  generation [table: data/VALIDATION_GATES.csv] -- _TODO(human): if not
  13/13, do not present this brief until they
  are; re-generate after fixing._

## Slide 3 -- Cumulative PCR curves and early-step effect

_TODO(human): pull the 2-4 most visually/statistically clear-cut PCR curve
panels from `figures/pcr_curves/`, and the corresponding early-step
(PCR@1/PCR@5) deltas from `data/pcr_paired_deltas_stepwise.csv`. Label any
selected "strongest" cue explicitly as a descriptive post-results
selection, not a preregistered primary comparison (spec section 9's
explicit requirement).*

## Slide 4 -- ResearchComplete/no-tool routes and trace raster

_TODO(human): one panel from `figures/route_decomposition/`, one raster from
`figures/trace_rasters/`._

## Slide 5 -- Evidence loss and recovery

_TODO(human): pull the clearest evidence-progression panel from
`figures/evidence/` and the paired-evidence-loss numbers for that same
comparison from `data/evidence_summary.csv`._

## Slide 6 -- Final-answer effect, limitations, and next decision

_TODO(human): pull from `figures/answers/` and `data/answer_paired_flips.csv`
(net_harm). State explicitly that these are pre-adjudication automatic
semantic labels (spec 3.3). State the next decision this lab meeting needs
to make (e.g., whether to proceed with Gemma/GPT-OSS integration once their
run matrices pass the same checks)._

# Missing or Incompatible Results

Every gap the task asked to be documented rather than papered over.

## 1. GPT-OSS-20B / 2WikiMultiHopQA — entirely missing

**0/108 shards complete** at export time. Root cause: a vLLM server-side
harmony-format parsing crash (`HarmonyError`) on certain malformed GPT-OSS
tool-call headers. A fix (`attack/runtime/gptoss_lenient_harmony_parser.py`)
was deployed and a rerun (SLURM job 166537, part of a 5-job batch also
covering Gemma27B's unrelated fix) was submitted and queued (`PD` /
pending-priority state) at export time, but had not started. **No
GPT-OSS/2WikiMultiHopQA number appears anywhere in this deck** — not
imputed, not estimated from the other 3 datasets. Every GPT-OSS macro-average
in this export is over 3 datasets; every Qwen/Llama macro-average is over 4.

## 2. GPT-OSS-20B — no dedicated local-completion evidence package

`analysis/evidence_hop_coverage/` (hop-stratification, paired bootstrap CI,
hand-validated fixtures) is explicitly scoped to
`CANONICAL_BUNDLE_MODELS = (qwen14b, llama8b)` only
(`build_evidence_hop_coverage.py:57`). No equivalent package has been built
for GPT-OSS. This export's GPT-OSS evidence numbers (`03_evidence/gpt_oss_evidence_mini_table.csv`)
were derived fresh, directly from `gemma_gptoss_results_bundle/data/evidence_run_level.csv`'s
`local_completion_coverage` column — the same underlying quantity, same
code path (`evidence_ext.py`, reused by both bundles) — via a simple paired
mean-difference, not the dedicated package's richer statistical treatment.
Real, comparable numbers; less rigorously validated than Qwen/Llama's. See
`03_evidence/evidence_results_notes.md`.

## 3. GPT-OSS-20B — no adjudication queue

Qwen/Llama's 543 ambiguous rows already have a blinded review packet built
(`qwen_llama_results_bundle/adjudication/`, not yet reviewed). GPT-OSS's own
flagged rows (visible per-condition in `04_answers/gpt_oss_answer_mini_table.csv`'s
`n_needs_adjudication` column) have not been queued for review at all. This
is future work, not an error in this export.

## 4. Gemma-3-27B-IT — excluded entirely, by explicit instruction

Not a gap in this export — deliberately out of scope. Its production data
is currently invalid (generation-time history-imitation bug, ~96% of rows
affected) and a corrected rerun is in progress (SLURM jobs 166533/166534/166535/166536).
`gemma_gptoss_results_bundle/` physically contains Gemma27B's files
alongside GPT-OSS's; this export's build scripts filter by an explicit
model-name token (not a bare filename-prefix glob) specifically to prevent
any Gemma file from being copied in — verified zero Gemma-named files
appear anywhere in the final ZIP.

## 5. Per-example (run-level) data — not included, by design, not by gap

`answer_run_level.csv`, `pcr_run_level.csv`, and `run_level_evidence_hops.csv`
were not copied into `05_appendix/`. For Qwen/Llama, `answer_run_level.csv`
is a git-LFS pointer stub in this working tree (not materialized locally);
for GPT-OSS it is a real ~75MB file. Neither was necessary: every table
this export produces is already derivable from the smaller run-level/summary
CSVs that were copied (`pcr_stepwise.csv`, `answer_summary.csv`,
`answer_paired_flips.csv`, `evidence_run_level.csv` for GPT-OSS specifically,
which at 524KB is small enough to include and was used directly). If a
future need arises for genuinely per-example detail, it remains at the
original canonical locations, untouched.

## 6. No new pooled/macro-averaged "model-level" PCR number existed before this export

`qwen_llama_results_bundle/MASTER_RESULTS.md` deliberately reports
per-dataset PCR ranges, not a single pooled number, arguing that pooling
would obscure real dataset-structure differences. This export's main-slide
PCR curves/mini-tables introduce a macro-average across datasets purely for
slide readability — a new, disclosed simplification for this deck, not a
change to the canonical numbers (which remain, unaveraged, in
`05_appendix/pcr/`). See `02_pcr/PCR_DEFINITION.md`'s aggregation-method
section for the full reasoning.

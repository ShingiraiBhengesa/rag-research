# Evidence Coverage Distributions v1

Distribution-level analysis of required-evidence coverage at the
researcher's first PCR-qualifying completion decision, for Qwen2.5-14B,
Llama-3.1-8B, and GPT-OSS-20B. Extends (does not redefine, recompute, or
compete with) the existing canonical evidence analysis in
`analysis/evidence_hop_coverage/` — this package answers a different
question ("what does the distribution of coverage values look like,
discretely and in full, per model x dataset x condition") than that
package's own paired-delta/hop-stratified summary focus, while sharing the
exact same underlying definitions and, where available, reusing its
output directly.

## Read this first

`SOURCE_AUDIT.md` — a genuine field-semantics problem was found and fixed
in GPT-OSS's source data before anything here was built. Read it before
trusting any GPT-OSS number in this package.

`EVIDENCE_COVERAGE_DISTRIBUTION_REPORT.md` — the self-contained narrative
report (all tables expanded inline, all figures embedded).

## Regenerating everything

```bash
python -m analysis.evidence_coverage_distributions_v1.build_evidence_distribution_figures --stage all
python -m unittest analysis.evidence_coverage_distributions_v1.test_evidence_distribution_figures -v
```

`--stage data` writes only the four CSVs + `validation_gates.csv`;
`--stage figures` additionally renders every PNG/PDF/SVG (reads canonical
sources fresh each time — nothing is cached between stages).

## Scope

- **Models**: Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct (4/4 datasets
  each); GPT-OSS-20B (3/4 — 2WikiMultiHopQA is incomplete, 38/108 shards as
  of 2026-10-06; a rerun is in progress but was not complete at analysis
  time. Never derived from a macro-average; simply absent).
- **Datasets**: HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHopRAG.
- **Canonical evidence variable**: required-evidence coverage at the
  researcher's first PCR-qualifying completion (`RESEARCH_COMPLETE` or a
  no-tool terminal exit) — never evidence gathered after that point. See
  `SOURCE_AUDIT.md` for the exact per-model source fields and the GPT-OSS
  fix.

## File map

| File | Phase | Contents |
|---|---|---|
| `SOURCE_AUDIT.md` | 1 | Source locations, field-level checks, the GPT-OSS boundary-mismatch finding and fix |
| `coverage_distribution_by_condition.csv` | 2 | eligible N, 4-category counts/percentages, mean/median/stdev/IQR coverage, mean/median missing, per model x dataset x condition |
| `missing_evidence_distribution.csv` | 5 | zero/one/two/three-plus-missing categories, overall + stratified by required-evidence count (2/3/4) where n >= 30 |
| `paired_evidence_loss_distribution.csv` | 6 | paired N, loss/no-change/gain percentages, mean/median paired loss, 95% paired-bootstrap CI (10,000 resamples, seed=0), per model x dataset x the 8 declared comparisons |
| `full_coverage_transitions.csv` | 7 | full-coverage 2x2 transition counts, loss/gain rates, exact McNemar p-value |
| `validation_gates.csv` | 11 | all 15 gates, PASS/FAIL + detail |
| `figures/slide/` | 3, 6 | one stacked-bar figure per model (natural + Oracle calibration), one cross-cutting diverging-bar figure |
| `figures/appendix/` | 4, 5 | ECDFs (primary + source-ablation, per model), missing-evidence stacked bars (per model) |
| `build_evidence_distribution_figures.py` | — | orchestrator (`--stage data\|figures\|all`) |
| `test_evidence_distribution_figures.py` | 11 | 35 unit tests |
| `source_loader.py`, `categories.py`, `missing_evidence.py`, `pairing.py`, `validation.py`, `writers.py`, `figures_*.py` | — | library modules the orchestrator calls |

## What this package deliberately does NOT do

- Redefine, recompute, or re-derive PCR. PCR's own terminal-route
  classification is read from canonical sources and used only as a filter
  (is this row a genuine completion at all), never recomputed.
- Write into `analysis/evidence_hop_coverage/`, any canonical results
  bundle, or any raw trace (enforced structurally — read-only access only
  — and checked by validation gate 14).
- Pool datasets in any primary table or figure (checked by gate 9) or
  substitute a macro-average for GPT-OSS's missing 2WikiMultiHopQA data
  (checked by gate 10).
- Use a violin plot or KDE for this bounded, discrete metric (checked by
  gate 11) — every distribution figure here is a plain histogram/stacked
  bar/ECDF of actually-observed values.

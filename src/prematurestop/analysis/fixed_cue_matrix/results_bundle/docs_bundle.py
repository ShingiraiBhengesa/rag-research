"""Narrative docs: `README.md`, `MASTER_RESULTS.md`, `CHAPTER5_WRITING_NOTES.md`,
`LAB_MEETING_BRIEF.md`.

Every number placed in these documents by this module is pulled directly
from an already-written `data/*.csv` (or `analysis_manifest.json`) and
carries an inline `[table: <path>]` citation, per spec section 10's gate 11
("every number in MASTER_RESULTS.md has a table/file reference") -- see
`gates.py::gate_11_cited_numbers`'s regex, which flags any line with a bare
digit and no `[table:` marker.

What this module deliberately does NOT generate: spec section 7.6's "main
findings and negative findings" prose, and the qualitative "short statement
of what is and is not supported" sentences in 7.2/7.3. Those require human
judgment applied to the verified numbers (see the plan doc's decision log)
-- this module leaves clearly marked `_TODO(human)_` placeholders for them,
each pointing at the specific cited tables/figures a human should read
before writing that paragraph. Filling those in is spec step 13's "final
claim-by-claim review," not a data-generation step.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from prematurestop.evaluation import config

from . import tables_bundle
from .comparisons import COMPARISONS


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt_pct(x, nd=1):
    if x in (None, "", "None"):
        return "NA"
    try:
        return f"{100.0 * float(x):.{nd}f}%"
    except (TypeError, ValueError):
        return str(x)


TODO_MARKER = "_TODO(human): "


def write_readme(bundle_root: Path) -> None:
    text = """# results/whitebox/e1_fixed_cue_matrix/qwen_llama

Frozen, verified results bundle for the completed `qwen14b`/`llama8b`
production runs (12 conditions x 4 datasets x 2 models). Built from
`results/traces/{qwen14b,llama8b}/` and `results/runs/{qwen14b,llama8b}/`
via `src/prematurestop/evaluation/completion.py` (the sole authoritative
PCR/route/eligibility source -- see
`results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md`).

## Regenerating everything

    conda activate whitebox
    python -m prematurestop.analysis.fixed_cue_matrix.results_bundle.build_bundle \\
        --bundle-root results/whitebox/e1_fixed_cue_matrix/qwen_llama --stage all

`--stage data` (default) runs steps 1-11 (freeze through validation gates)
and writes everything under `data/`. `--stage figures`/`--stage tables`/
`--stage workbook` consume `data/*.csv` only and can be re-run independently
once `data/` exists. `--stage all` runs all four in order.

No file in this bundle is hand-edited except the narrative sections marked
`_TODO(human)_` in `MASTER_RESULTS.md` -- everything else, including every
number in every `.md`/`.tex` file and every figure, is generated from
`data/*.csv` and will be silently overwritten by a re-run. If you edit a
number by hand, the next `--stage all` run erases the edit; fix the
generator in `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/`, not the output file.

## Directory guide

- `data/` -- the 13 canonical CSVs (spec section 5) plus 2 supporting files
  (`mechanism_run_level.csv`/`mechanism_summary.csv`, feeding the trace-raster
  figures only -- not part of the spec's required 13).
- `tables/` -- Markdown tables generated from `data/`. Start at
  `tables/TABLE_INDEX.md` for sign conventions and file-join keys before
  citing any number.
- `figures/` -- PNG+PDF+caption triples, one per figure, organized by type.
- `appendix/` -- the same numbers as LaTeX booktabs tables, for direct
  inclusion in the thesis appendix.
- `qwen_llama_results_workbook.xlsx` -- one sheet per `data/*.csv`.
- `analysis_manifest.json` -- the data freeze: git commit, input file
  hashes, shard completion, active-job check (must be empty for these two
  models).
- `data/VALIDATION_GATES.csv` -- the 13 validation gates (spec section 10);
  the bundle is not final until every row here is PASS.

## Locked scope and definitions

Models: qwen14b, llama8b only. Datasets: HotpotQA, 2WikiMultiHopQA,
MuSiQue, MultiHopRAG. All 12 production conditions. PCR = **Post-observation
Completion Rate**, never "Premature" -- see `MASTER_RESULTS.md` section 1
for the full locked-definitions text. Gemma27b and GPT-OSS-20B are excluded
from this bundle until their run matrices pass the same completion/
eligibility checks; legacy Attack-A/WGAR/Cue-Only/Bridge-Fact results live
under `archive/legacy_results/`, never merged here (gate 13 checks this).
"""
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "README.md").write_text(text, encoding="utf-8")


def write_chapter5_writing_notes(bundle_root: Path, data_dir: Path) -> None:
    lines = [
        "# Chapter 5 writing notes",
        "",
        "Maps spec section 8's \"Chapter 5 presentation\" requirements to the files "
        "that already satisfy them, and embeds the actual tables inline below so "
        "nothing requires jumping to a separate file to check a number.",
        "",
        "## Main chapter",
        "",
        "| Requirement | Source |",
        "|---|---|",
        "| Cumulative PCR curves, every step | `figures/pcr_curves/*.png` (main = 5 prespecified conditions; `figures/pcr_curves/appendix/` = all 12) |",
        "| Compact exact-value tables PCR@{1,5,10,30} | embedded below, and `tables/main_chapter_tables.md` |",
        "| Route decomposition summary | `figures/route_decomposition/*.png`, embedded below |",
        "| 1-2 paired trace rasters per model | `figures/trace_rasters/*.png` |",
        "| Evidence coverage/loss summary | `figures/evidence/*.png`, embedded below |",
        "| Paired answer-flip summary | `figures/answers/*.png`, embedded below |",
        "| Retrieval-equivalence sensitivity summary | embedded below, and `data/retrieval_equivalence.csv` |",
        "",
        "## Appendix",
        "",
        "| Requirement | Source |",
        "|---|---|",
        "| Full PCR values, every step/condition | `appendix/full_pcr_tables.tex`, embedded below |",
        "| Full paired PCR deltas, every step | `appendix/full_pcr_tables.tex` (second section), `data/pcr_paired_deltas_stepwise.csv` |",
        "| Full route-specific cumulative values | `appendix/full_route_tables.tex`, embedded below |",
        "| Complete evidence and answer tables | `appendix/evidence_tables.tex`, `appendix/answer_tables.tex`, embedded below |",
        "",
        "The appendix tables and the main-chapter curves are generated from the same "
        "stepwise CSVs (`pcr_stepwise.csv`, `pcr_paired_deltas_stepwise.csv`, "
        "`route_decomposition_stepwise.csv`) -- they cannot disagree by construction.",
        "",
        "## Before writing prose",
        "",
        "1. Read `MASTER_RESULTS.md` in full, including every `_TODO(human)_` block.",
        "2. Check `data/VALIDATION_GATES.csv` -- do not write Chapter 5 text against "
        "a bundle with a failing gate.",
        "3. Every number you write in the thesis should trace back to a specific "
        "cell in one of the `data/*.csv` files or a specific figure/table file -- "
        "if you cannot point to the source, do not write the number.",
        "4. Answer accuracy numbers are pre-adjudication (spec 3.3) until the "
        "blinded review of `needs_adjudication_flag == True` rows is complete -- "
        "say so explicitly wherever an accuracy number appears, not only in a footnote.",
        "",
        "## All tables (embedded, both models)",
        "",
        "### Compact PCR@k tables, all 12 conditions [table: data/pcr_stepwise.csv]",
        "",
        tables_bundle.build_main_chapter_tables_md(data_dir),
        "### Retrieval-equivalence sensitivity summary [table: data/retrieval_equivalence.csv]",
        "",
        tables_bundle.build_retrieval_equivalence_summary_md(data_dir),
        "### Route decomposition -- final terminal-route tallies [table: data/route_decomposition_stepwise.csv]",
        "",
        tables_bundle.build_full_route_tables_md(data_dir),
        "",
        "### Evidence coverage and paired evidence loss (final-report-input boundary) [table: data/evidence_summary.csv]",
        "",
        "This is coverage of what `final_report_generation` actually received "
        "(citation-based, downstream of any redelegation/compression) -- a "
        "DIFFERENT boundary from `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage`'s "
        "local-completion-at-T_i paired evidence loss. See "
        "`tables/TABLE_INDEX.md`'s boundary-disambiguation section before "
        "citing either number.",
        "",
        tables_bundle.build_evidence_tables_md(data_dir),
        "",
        "### Answer accuracy and paired flips, pre-adjudication [table: data/answer_summary.csv, data/answer_paired_flips.csv]",
        "",
        tables_bundle.build_answer_tables_md(data_dir),
        "",
        "### Full stepwise PCR, every step 1-30, all 12 conditions, both models (values in %) [table: data/pcr_stepwise.csv]",
        "",
        "Column codes:",
        "",
        tables_bundle.condition_code_legend_md(),
        "",
        tables_bundle.build_full_pcr_tables_md(data_dir),
    ]
    (bundle_root / "CHAPTER5_WRITING_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lab_meeting_brief(bundle_root: Path, data_dir: Path) -> None:
    run_inv = _read_csv(data_dir / "run_inventory.csv")
    total_valid = sum(int(r["n_valid"]) for r in run_inv) if run_inv else 0
    gates_rows = _read_csv(data_dir / "VALIDATION_GATES.csv")
    n_pass = sum(1 for r in gates_rows if r.get("status") == "PASS")

    text = f"""# Lab meeting brief

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
- 12 conditions x 4 datasets x 2 models, {total_valid} valid completed runs
  total [table: data/run_inventory.csv].
- Validation gates: {n_pass}/{len(gates_rows)} passing as of this brief's
  generation [table: data/VALIDATION_GATES.csv] -- {TODO_MARKER}if not
  {len(gates_rows)}/{len(gates_rows)}, do not present this brief until they
  are; re-generate after fixing._

## Slide 3 -- Cumulative PCR curves and early-step effect

{TODO_MARKER}pull the 2-4 most visually/statistically clear-cut PCR curve
panels from `figures/pcr_curves/`, and the corresponding early-step
(PCR@1/PCR@5) deltas from `data/pcr_paired_deltas_stepwise.csv`. Label any
selected "strongest" cue explicitly as a descriptive post-results
selection, not a preregistered primary comparison (spec section 9's
explicit requirement).*

## Slide 4 -- ResearchComplete/no-tool routes and trace raster

{TODO_MARKER}one panel from `figures/route_decomposition/`, one raster from
`figures/trace_rasters/`._

## Slide 5 -- Evidence loss and recovery

{TODO_MARKER}pull the clearest evidence-progression panel from
`figures/evidence/` and the paired-evidence-loss numbers for that same
comparison from `data/evidence_summary.csv`._

## Slide 6 -- Final-answer effect, limitations, and next decision

{TODO_MARKER}pull from `figures/answers/` and `data/answer_paired_flips.csv`
(net_harm). State explicitly that these are pre-adjudication automatic
semantic labels (spec 3.3). State the next decision this lab meeting needs
to make (e.g., whether to proceed with Gemma/GPT-OSS integration once their
run matrices pass the same checks)._
"""
    (bundle_root / "LAB_MEETING_BRIEF.md").write_text(text, encoding="utf-8")


def _data_freeze_section(bundle_root: Path, data_dir: Path) -> list[str]:
    manifest_path = bundle_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    run_inv = _read_csv(data_dir / "run_inventory.csv")

    lines = [
        "## 7.1 Data freeze and validity",
        "",
        f"- Git commit: `{manifest.get('git_commit', 'unknown')}` [table: analysis_manifest.json]",
        f"- Production root: `{manifest.get('production_root', '')}` [table: analysis_manifest.json]",
        f"- Models included: {', '.join(manifest.get('included_models', []))} "
        f"[table: analysis_manifest.json]",
        f"- Active SLURM jobs for included models at freeze time: "
        f"{len(manifest.get('active_slurm_jobs_for_included_models', []))} "
        f"[table: analysis_manifest.json] -- must be 0; if not, the bundle is not frozen.",
        "",
        "### Planned / completed / valid / eligible counts by cell",
        "",
        "See `data/run_inventory.csv` for the full 96-cell "
        "(2 models x 4 datasets x 12 conditions) breakdown "
        "[table: data/run_inventory.csv]. Totals:",
        "",
    ]
    if run_inv:
        n_planned = sum(int(r["n_planned"]) for r in run_inv)
        n_completed = sum(int(r["n_completed"]) for r in run_inv)
        n_valid = sum(int(r["n_valid"]) for r in run_inv)
        n_pcr_eligible = sum(int(r["n_pcr_eligible"]) for r in run_inv)
        n_evidence_eligible = sum(int(r["n_evidence_eligible"]) for r in run_inv)
        n_answer_eligible = sum(int(r["n_answer_eligible"]) for r in run_inv)
        n_failures = sum(int(r["n_failures"]) for r in run_inv)
        lines += [
            f"- Planned: {n_planned} [table: data/run_inventory.csv]",
            f"- Completed: {n_completed} [table: data/run_inventory.csv]",
            f"- Valid: {n_valid} [table: data/run_inventory.csv]",
            f"- PCR-eligible: {n_pcr_eligible} [table: data/run_inventory.csv]",
            f"- Evidence-eligible: {n_evidence_eligible} [table: data/run_inventory.csv]",
            f"- Answer-eligible: {n_answer_eligible} [table: data/run_inventory.csv]",
            f"- Failures (technical/parser/logging): {n_failures} [table: data/run_inventory.csv]",
            "",
        ]
    lines += [
        f"Metric/code version: `canonical_completion.py` (authoritative PCR/route source, see "
        f"`results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md`), git commit above.",
        "",
    ]
    return lines


def _per_model_paired_deltas_md(data_dir: Path, model: str) -> str:
    """Compact PCR@{1,5,10,30} paired-delta table (all 14 computed
    comparisons x 4 datasets) for this model, pulled straight from
    `pcr_paired_deltas_stepwise.csv` -- the same file
    `tables_bundle.build_full_pcr_tables_tex`'s appendix delta section
    reads, just reformatted as Markdown here."""
    rows = _read_csv(data_dir / "pcr_paired_deltas_stepwise.csv")
    rows = [r for r in rows if r["model"] == model and r["step"] in ("1", "5", "10", "30")]
    by_key: dict[tuple, dict[str, str]] = {}
    for r in rows:
        by_key.setdefault((r["dataset"], r["comparison_name"], r["n_paired"]), {})[r["step"]] = r["delta_pcr"]
    lines = [
        "| Dataset | Comparison | n_paired | ΔPCR@1 | ΔPCR@5 | ΔPCR@10 | ΔPCR@30 |",
        "|---|---|---|---|---|---|---|",
    ]
    for (dataset, comparison_name, n_paired), by_step in sorted(by_key.items()):
        disp_ds = config.DISPLAY_DATASET_NAMES.get(dataset, dataset)
        vals = " | ".join(_fmt_pct(by_step.get(k)) for k in ("1", "5", "10", "30"))
        lines.append(f"| {disp_ds} | {comparison_name} | {n_paired} | {vals} |")
    return "\n".join(lines)


def _per_model_section(section_num: str, model: str, data_dir: Path) -> list[str]:
    lines = [f"## {section_num} {config.DISPLAY_MODEL_NAMES.get(model, model)} results", ""]

    lines += ["### Compact PCR@k tables (all 12 conditions) [table: data/pcr_stepwise.csv]", ""]
    lines.append(tables_bundle.build_main_chapter_tables_md(data_dir, model=model))

    lines += ["### Paired PCR deltas at PCR@{1,5,10,30}, all comparisons [table: data/pcr_paired_deltas_stepwise.csv]", ""]
    lines.append(_per_model_paired_deltas_md(data_dir, model))
    lines.append("")

    lines += [
        "### Full stepwise PCR (every step 1-30, all 12 conditions) [table: data/pcr_stepwise.csv]",
        "",
        "Values in %. Column codes:",
        "",
        tables_bundle.condition_code_legend_md(),
        "",
    ]
    lines.append(tables_bundle.build_full_pcr_tables_md(data_dir, model=model))

    lines += ["### Route decomposition -- final terminal-route tallies, all 7 canonical routes [table: data/route_decomposition_stepwise.csv]", ""]
    lines.append(tables_bundle.build_full_route_tables_md(data_dir, model=model))
    lines.append("")

    lines += [
        "### Evidence coverage and paired evidence loss (final-report-input boundary) [table: data/evidence_summary.csv]",
        "",
        "This is coverage of what `final_report_generation` actually received "
        "(citation-based, downstream of any redelegation/compression) -- a "
        "DIFFERENT boundary from `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage`'s "
        "local-completion-at-T_i paired evidence loss (see "
        "`tables/TABLE_INDEX.md`).",
        "",
    ]
    evidence_md = tables_bundle.build_evidence_tables_md(data_dir, model=model)
    lines.append(evidence_md if evidence_md else "_No evidence rows for this model._")
    lines.append("")

    lines += [
        "### Answer accuracy and paired flips (pre-adjudication) [table: data/answer_summary.csv, data/answer_paired_flips.csv]",
        "",
        "Automatic semantic labels, not adjudicated accuracy (spec 3.3) -- see `needs_adjudication_flag` "
        "counts in `data/answer_run_level.csv` before citing.",
        "",
    ]
    lines.append(tables_bundle.build_answer_tables_md(data_dir, model=model))
    lines.append("")

    lines += ["### Retrieval-equivalence sensitivity summary [table: data/retrieval_equivalence.csv]", ""]
    lines.append(tables_bundle.build_retrieval_equivalence_summary_md(data_dir, model=model))

    for dataset in config.DATASETS:
        lines += [
            f"### {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)} -- narrative summary",
            "",
            f"{TODO_MARKER}cumulative PCR behavior (see `figures/pcr_curves/pcr_curve_{model}_{dataset}_main.png` "
            f"and the full stepwise table above), largest early-step paired differences (paired-deltas table "
            f"above, filtered to dataset={dataset}, PCR@1/PCR@5), ResearchComplete-vs-no-tool route mix "
            f"(`figures/route_decomposition/termination_routes_{model}.png` and the route table above), "
            f"search/thinking/recovery changes (`data/process_metrics.csv`), evidence coverage/loss (evidence "
            f"table above, filtered to dataset={dataset}), answer accuracy and paired flips (answer table above, "
            f"filtered to dataset={dataset}), identical-observation sensitivity result (retrieval-equivalence "
            f"table above), and a short statement of what is and is not supported by this dataset's numbers._",
            "",
        ]
    return lines


def write_master_results(bundle_root: Path, data_dir: Path) -> None:
    lines = [
        "# MASTER_RESULTS.md",
        "",
        "Every number below cites the `data/*.csv` file it was read from -- "
        "see `tables/TABLE_INDEX.md` for sign conventions before reading any delta.",
        "",
        "## Locked metric definitions",
        "",
        "PCR means **Post-observation Completion Rate**, never Premature Completion Rate. "
        "For each PCR-eligible run, T_i is the first post-observation researcher decision "
        "that ends the targeted episode through RESEARCH_COMPLETE or NO_TOOL_DIRECT_EXIT. "
        "PCR@k is the cumulative fraction of eligible runs with T_i <= k; one researcher-node "
        "decision (including think_tool) is one step; the full curve uses steps 1-29, PCR@30 "
        "is the end-of-budget summary. The raw DIRECT_ANSWER implementation label is displayed "
        "as no-tool exit throughout this bundle, never as proof of a concise answer.",
        "",
        "Evidence deficit = 1 - coverage. Paired evidence loss = control coverage minus cue "
        "coverage (`data/evidence_summary.csv::paired_evidence_loss`) -- note this is the "
        "*opposite* sign direction from `data/pcr_paired_deltas_stepwise.csv::delta_pcr` "
        "(cue minus control); both independently mean \"positive = attack made it worse.\" "
        "See `tables/TABLE_INDEX.md`.",
        "",
        "**This bundle's evidence-loss numbers measure final-report-input coverage** "
        "(citation-based, what `final_report_generation` actually received) -- a "
        "different, downstream boundary from `src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage`'s "
        "local-completion-at-T_i evidence loss. The two are not interchangeable "
        "and will report different numbers for the same comparison by design "
        "(coverage at local completion and at final-run termination are separate "
        "outcomes); see `tables/TABLE_INDEX.md` for the full boundary "
        "disambiguation and a per-row consistency cross-check between the two "
        "packages.",
        "",
        "Main downstream answer measure is semantic Answer Accuracy with adjudication status "
        "visible (`needs_adjudication_flag`); automatic semantic labels are not adjudicated "
        "accuracy until the blinded review is complete (spec 3.3).",
        "",
    ]
    lines += _data_freeze_section(bundle_root, data_dir)
    lines += _per_model_section("7.2", "qwen14b", data_dir)
    lines += _per_model_section("7.3", "llama8b", data_dir)
    lines += [
        "## 7.4 Direct model comparison",
        "",
        f"{TODO_MARKER}compare qwen14b and llama8b on baseline completion trajectory, cue-induced "
        f"PCR shift, route composition, search reduction, supervisor recovery, evidence loss, "
        f"final-answer degradation, and technical/tool-calling failure behavior -- cite "
        f"`data/pcr_stepwise.csv`, `data/pcr_paired_deltas_stepwise.csv`, "
        f"`data/route_decomposition_stepwise.csv`, `data/process_metrics.csv`, "
        f"`data/evidence_summary.csv`, `data/answer_summary.csv` for each claim. Do not compare "
        f"raw percentages across models without first showing their different baseline rates and "
        f"denominators (`data/run_inventory.csv`)._",
        "",
        "## 7.5 Dataset comparison",
        "",
        f"{TODO_MARKER}discuss HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG separately "
        f"before any pooled statement -- note differences in corpus size, distractors, hop count, "
        f"and evidence representation. Cite the per-dataset sections in 7.2/7.3 above._",
        "",
        "## 7.6 Main findings and negative findings",
        "",
        f"{TODO_MARKER}this section MUST include both positive and negative results. A cue that "
        f"changes PCR but not evidence or answer accuracy is a process-level effect, not proven "
        f"downstream harm -- check `data/evidence_summary.csv` and `data/answer_paired_flips.csv` "
        f"before claiming harm from a PCR effect alone. A flat final-answer result must not be "
        f"hidden. A flat PCR result with elevated parser or malformed-exit failures "
        f"(`data/route_decomposition_stepwise.csv`'s `final_n_TOOL_OR_PARSER_FAILURE`/"
        f"`final_n_TRACE_LOGGING_FAILURE`) is not evidence of clean robustness -- check those "
        f"counts before writing a robustness claim. Do not write this section until "
        f"`data/VALIDATION_GATES.csv` is all-PASS and sections 7.2-7.5 above are complete._",
        "",
    ]
    (bundle_root / "MASTER_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_docs(bundle_root: Path) -> None:
    data_dir = bundle_root / "data"
    write_readme(bundle_root)
    write_chapter5_writing_notes(bundle_root, data_dir)
    write_master_results(bundle_root, data_dir)
    write_lab_meeting_brief(bundle_root, data_dir)

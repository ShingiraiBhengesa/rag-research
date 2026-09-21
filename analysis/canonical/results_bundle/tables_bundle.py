"""Step 12 (tables): Markdown tables + LaTeX appendix tables.

`analysis/final_paper_evaluation/paper_tables.py`'s LaTeX generators are
built against the legacy `pcr.py` checkpoint-CSV column names
(`pcr@1`, `delta_pcr@3_vs_neutral`, `pct_research_complete`, ...) and can't
be called unchanged against this bundle's new stepwise schema -- their
*conventions* (booktabs style, `_fmt`/`_esc` helpers) are copied here,
reading the new canonical CSVs directly, so no numbers are recomputed, only
reformatted. Unlike `paper_tables.py` (one file per table type per model),
the spec's required `appendix/` directory (section 4) lists exactly one file
per table type covering both models -- each `.tex` file below loops over
the bundle's `CANONICAL_BUNDLE_MODELS` internally instead of being split per model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from analysis import config

from .comparisons import COMPARISONS
from .scope import CANONICAL_BUNDLE_MODELS


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt(x, nd=1, pct=False):
    if x is None or x in ("", "None", "NA"):
        return "NA"
    try:
        v = float(x)
        if pct:
            v *= 100.0
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


CHAPTER5_CHECKPOINTS = (1, 5, 10, 30)

# Short column codes for the wide (12-condition) full-stepwise tables only.
# `config.CONDITION_DISPLAY_NAMES` values like "Epistemic confirmation +
# source" and "PreRegisteredSufficiencySaturation" make a 12-column table's
# header row so wide it soft-wraps mid-word in most editors, which reads as
# a garbled/broken table even though the underlying step values (always
# capped at 30, never beyond -- see gate 3) are correct. A short-code legend
# is printed once above each such table instead.
_SHORT_CONDITION_CODES: dict[str, str] = {
    "NaturalClean": "Clean",
    "NaturalNeutral": "Neutral",
    "NaturalContinueSearch": "ContSearch",
    "Natural_epistemic_confirmation": "EpistConf",
    "Natural_search_match_cue": "SearchMatch",
    "Natural_source_coverage": "SrcCov",
    "Natural_epistemic_confirmation_source": "EpistConf+Src",
    "Natural_search_match_source": "SearchMatch+Src",
    "Natural_PreRegisteredSufficiency": "PreRegSuff",
    "Natural_PreRegisteredSufficiencySaturation": "PreRegSuffSat",
    "OracleComplete_Neutral": "OracleNeutral",
    "OracleComplete_PreRegisteredSufficiencySaturation": "OraclePreRegSuffSat",
}


def condition_code_legend_md() -> str:
    lines = ["| Code | Condition |", "|---|---|"]
    for cond in config.CONDITIONS:
        lines.append(f"| {_SHORT_CONDITION_CODES.get(cond, cond)} | {config.CONDITION_DISPLAY_NAMES.get(cond, cond)} |")
    return "\n".join(lines)


def _pcr_at_checkpoints(stepwise_rows: list[dict]) -> dict[tuple[str, str, str], dict[int, float]]:
    by_cell: dict[tuple[str, str, str], dict[int, float]] = {}
    for r in stepwise_rows:
        step = int(r["step"])
        if step not in CHAPTER5_CHECKPOINTS:
            continue
        key = (r["model"], r["dataset"], r["condition"])
        pcr = r["pcr"]
        by_cell.setdefault(key, {})[step] = float(pcr) if pcr not in ("", "None") else None
    return by_cell


def _retrieval_equivalence_summary_rows(data_dir: Path) -> list[dict]:
    """Aggregates `retrieval_equivalence.csv` per (model, comparison) into
    the compact "sensitivity summary" spec section 8's main chapter asks
    for -- pct of readable pairs with identical retrieval, and the stricter
    identical-observation-sensitivity subset's own denominator (spec 5.6)."""
    rows = _read_csv(data_dir / "retrieval_equivalence.csv")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["model"], f"{r['baseline']} vs {r['condition']}"), []).append(r)
    out = []
    for (model, comparison), rs in sorted(grouped.items()):
        readable = [r for r in rs if r["both_found"] == "True"]
        n_readable = len(readable)
        n_identical_doc_set = sum(1 for r in readable if r["identical_doc_id_set"] == "True")
        n_sensitivity_pop = sum(1 for r in rs if r["identical_observation_sensitivity_population"] == "True")
        out.append(
            {
                "model": model, "comparison": comparison,
                "n_readable_pairs": n_readable,
                "pct_identical_doc_set": (100.0 * n_identical_doc_set / n_readable) if n_readable else None,
                "n_identical_observation_sensitivity_population": n_sensitivity_pop,
                "n_total_pairs": len(rs),
            }
        )
    return out


def build_main_chapter_tables_md(data_dir: Path, model: str | None = None) -> str:
    """Returns the Markdown body (no leading `# Main chapter tables` title
    -- callers that embed this inline, e.g. `MASTER_RESULTS.md`, supply
    their own heading level). `model=None` covers both models (used by
    `write_main_chapter_tables_md`); passing a single model restricts to it
    (used when embedding into that model's `MASTER_RESULTS.md` section)."""
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    checkpoints = _pcr_at_checkpoints(stepwise)
    run_inv = {(r["model"], r["dataset"], r["condition"]): r for r in _read_csv(data_dir / "run_inventory.csv")}
    models = (model,) if model else CANONICAL_BUNDLE_MODELS

    lines = []
    for m in models:
        lines.append(f"## {config.DISPLAY_MODEL_NAMES.get(m, m)}")
        for dataset in config.DATASETS:
            lines.append(f"### {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}")
            lines.append("")
            header = "| Condition | n_eligible | " + " | ".join(f"PCR@{k}" for k in CHAPTER5_CHECKPOINTS) + " |"
            sep = "|---|---|" + "|".join(["---"] * len(CHAPTER5_CHECKPOINTS)) + "|"
            lines += [header, sep]
            for cond in config.CONDITIONS:
                key = (m, dataset, cond)
                cps = checkpoints.get(key, {})
                inv = run_inv.get(key)
                n_elig = inv["n_pcr_eligible"] if inv else "?"
                disp = config.CONDITION_DISPLAY_NAMES.get(cond, cond)
                row = f"| {disp} | {n_elig} | " + " | ".join(_fmt(cps.get(k), pct=True) for k in CHAPTER5_CHECKPOINTS) + " |"
                lines.append(row)
            lines.append("")
    return "\n".join(lines)


def build_retrieval_equivalence_summary_md(data_dir: Path, model: str | None = None) -> str:
    lines = [
        "Main result is the same-example end-to-end comparison (`n_readable_pairs` denominator); "
        "`n_identical_observation_sensitivity_population` is the stricter subset where retrieval was "
        "byte-identical before any cue suffix, with its own denominator (spec 5.6) -- see "
        "`data/retrieval_equivalence.csv` for per-pair detail.",
        "",
    ]
    re_rows = _retrieval_equivalence_summary_rows(data_dir)
    if model:
        re_rows = [r for r in re_rows if r["model"] == model]
    if re_rows:
        lines.append(pd.DataFrame(re_rows).to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def write_main_chapter_tables_md(data_dir: Path, path: Path) -> None:
    lines = ["# Main chapter tables", "", "Compact PCR@k tables (spec section 8) -- full step-1-29 values are in `full_pcr_tables.md`.", ""]
    lines.append(build_main_chapter_tables_md(data_dir))
    lines += ["## Retrieval-equivalence sensitivity summary", ""]
    lines.append(build_retrieval_equivalence_summary_md(data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_full_pcr_tables_md(data_dir: Path, model: str | None = None) -> str:
    """Every PCR value, every step 1-30, formatted as a percentage. Steps
    are cast to `int` before pivoting -- `pcr_stepwise.csv`'s `step` column
    is read back as a string by `csv.DictReader`, and a plain
    `pivot_table(index="step", ...)` on string steps sorts lexically
    ("1","10","11",...,"2","20",...) rather than numerically; this produced
    a genuinely out-of-order table in an earlier version of this file."""
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    df = pd.DataFrame(stepwise)
    lines = []
    if not df.empty:
        df = df.copy()
        df["step"] = df["step"].astype(int)
        df["pcr"] = pd.to_numeric(df["pcr"], errors="coerce")
        models = (model,) if model else CANONICAL_BUNDLE_MODELS
        for m in models:
            for dataset in config.DATASETS:
                sub = df[(df["model"] == m) & (df["dataset"] == dataset)]
                if sub.empty:
                    continue
                lines.append(f"## {config.DISPLAY_MODEL_NAMES.get(m, m)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}")
                pivot = sub.pivot_table(index="step", columns="condition", values="pcr", aggfunc="first").sort_index()
                pivot.columns = [_SHORT_CONDITION_CODES.get(c, c) for c in pivot.columns]
                pivot = (pivot * 100).round(1)
                lines.append(pivot.to_markdown())
                lines.append("")
    return "\n".join(lines)


def write_full_pcr_tables_md(data_dir: Path, path: Path) -> None:
    lines = [
        "# Full PCR tables (every step 1-29, plus PCR@30, all values in %)",
        "",
        "Column codes (12 conditions -- see `main_chapter_tables.md` for full names):",
        "",
        condition_code_legend_md(),
        "",
    ]
    lines.append(build_full_pcr_tables_md(data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_full_route_tables_md(data_dir: Path, model: str | None = None) -> str:
    rows = _read_csv(data_dir / "route_decomposition_stepwise.csv")
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    final_cols = [c for c in df.columns if c.startswith("final_n_")]
    if model:
        df = df[df["model"] == model]
    last_per_cell = df.drop_duplicates(subset=["model", "dataset", "condition"], keep="last").copy()
    last_per_cell["condition"] = last_per_cell["condition"].map(lambda c: config.CONDITION_DISPLAY_NAMES.get(c, c))
    last_per_cell["model"] = last_per_cell["model"].map(lambda m: config.DISPLAY_MODEL_NAMES.get(m, m))
    last_per_cell["dataset"] = last_per_cell["dataset"].map(lambda d: config.DISPLAY_DATASET_NAMES.get(d, d))
    return last_per_cell[["model", "dataset", "condition"] + final_cols].to_markdown(index=False)


def write_full_route_tables_md(data_dir: Path, path: Path) -> None:
    lines = ["# Full route-decomposition tables (final terminal-route tallies, all 7 canonical routes)", ""]
    lines.append(build_full_route_tables_md(data_dir))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_evidence_tables_md(data_dir: Path, model: str | None = None) -> str:
    summary_rows = _read_csv(data_dir / "evidence_summary.csv")
    if model:
        summary_rows = [r for r in summary_rows if r["model"] == model]
    if not summary_rows:
        return ""
    return pd.DataFrame(summary_rows).to_markdown(index=False)


def write_evidence_tables_md(data_dir: Path, path: Path) -> None:
    lines = [
        "# Evidence tables", "",
        "Paired evidence loss = control coverage - cue coverage (spec 3.2 sign convention).",
        "",
        "**This measures final-report-input coverage** (citation-based, what "
        "`final_report_generation` actually received) -- a DIFFERENT, "
        "downstream boundary from `analysis/evidence_hop_coverage`'s "
        "local-completion-at-T_i paired evidence loss. The two are not "
        "interchangeable; see `TABLE_INDEX.md`'s \"Which evidence coverage "
        "boundary is this?\" section before citing either.",
        "",
    ]
    lines.append(build_evidence_tables_md(data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_answer_tables_md(data_dir: Path, model: str | None = None) -> str:
    summary_rows = _read_csv(data_dir / "answer_summary.csv")
    flips_rows = _read_csv(data_dir / "answer_paired_flips.csv")
    if model:
        summary_rows = [r for r in summary_rows if r["model"] == model]
        flips_rows = [r for r in flips_rows if r["model"] == model]
    lines = []
    if summary_rows:
        lines.append("**Per-cell summary**")
        lines.append("")
        lines.append(pd.DataFrame(summary_rows).to_markdown(index=False))
        lines.append("")
    if flips_rows:
        lines.append("**Paired answer flips**")
        lines.append("")
        lines.append(pd.DataFrame(flips_rows).to_markdown(index=False))
    return "\n".join(lines)


def write_answer_tables_md(data_dir: Path, path: Path) -> None:
    lines = ["# Answer tables", "", "**Pre-adjudication**: `semantic_correct`/`pct_semantic_correct` are automatic labels; see `needs_adjudication_flag` counts before citing as final accuracy (spec 3.3).", ""]
    lines.append(build_answer_tables_md(data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_index_md(path: Path) -> None:
    text = """# Table index

All tables below are generated from the canonical CSVs in `data/`, never
hand-typed. Regenerate everything with:

    python -m analysis.canonical.results_bundle.build_bundle --stage data
    python -m analysis.canonical.results_bundle.build_bundle --stage tables

## Sign conventions (read before citing any delta)

- `pcr_paired_deltas_stepwise.csv::delta_pcr` = `cue_pcr - control_pcr`.
  Positive = the attack condition completed MORE by step k (attack
  accelerated completion).
- `evidence_summary.csv::paired_evidence_loss` = `control_mean_coverage -
  cue_mean_coverage`. Positive = the cue condition lost evidence relative to
  control (spec 3.2's literal wording -- the OPPOSITE sign direction from
  `delta_pcr` above; both independently mean "positive = attack made it
  worse").

## Which "evidence coverage" boundary is this? (read before comparing across packages)

`evidence_summary.csv`'s `control_mean_coverage`/`cue_mean_coverage` are
built from `EvidenceRecord.final_input_coverage_fraction` --
**final-report-input coverage**: citation-based coverage of what
`final_report_generation` actually received, extracted from that node's own
raw input text (see `analysis/evidence.py`'s module docstring, boundary 4).
This is a **downstream** boundary, after any supervisor redelegation and
after whatever the compression/synthesis step preserved.

This is a **different, not-directly-comparable** number from
`analysis/evidence_hop_coverage/paired_evidence_deltas.csv`'s
`paired_evidence_loss`, which is built from `local_coverage_fraction` --
**local-completion coverage at T_i**: raw retrieved-paragraph-ID coverage at
the researcher's own first post-observation completion decision (the same
`T_i` this repo's PCR numbers use), computed by ID-set intersection, never
by citation-text parsing. This is an **upstream** boundary, before any
redelegation or compression.

Both are legitimate, both are "paired evidence loss," and for the same
`(model, dataset, comparison)` cell they will report **different numbers**
-- this is expected, not a bug (spec 3.2: "coverage at local completion and
final-run coverage are separate outcomes"). The two packages' underlying
per-row retrieval computation was cross-checked and found consistent (>99%
exact per-row agreement between this package's own
`evidence_run_level.csv::local_completion_coverage` and
`evidence_hop_coverage`'s `local_coverage_fraction` wherever both are
defined -- the remaining <1% is `evidence_hop_coverage` correctly reporting
`NA` for `MAX_BUDGET_EXIT` rows that this package's broader field does not
gate on route). See `analysis/evidence_hop_coverage/README.md`'s
reconciliation section for the full cross-check.

**Rule of thumb**: cite `evidence_summary.csv` for "did the final answer
generator actually see the evidence." Cite
`analysis/evidence_hop_coverage/paired_evidence_deltas.csv` for "had the
researcher retrieved the evidence by the time it decided to stop
searching." Never quote one as if it were the other.

## `route_decomposition_stepwise.csv` vs `pcr_stepwise.csv`

These two files do not repeat each other's columns. `pcr_stepwise.csv` has
`pcr`/`n_total`/`n_eligible`/route-specific PCR percentages per step.
`route_decomposition_stepwise.csv` has only the raw route-specific counts
per step plus final route tallies. Join on `(model, dataset, condition,
step)` to combine them.

## Files

- `main_chapter_tables.md` -- compact PCR@{1,5,10,30} tables (spec section 8).
- `full_pcr_tables.md` -- every PCR value, every step 1-29 plus 30, every condition.
- `full_route_tables.md` -- final terminal-route tallies for all 7 canonical routes.
- `evidence_tables.md` -- paired evidence-loss summary.
- `answer_tables.md` -- per-cell answer summary + paired answer-flip table.
- `../appendix/*.tex` -- the same underlying numbers as LaTeX booktabs tables.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_full_pcr_tables_tex(data_dir: Path) -> str:
    """`appendix/full_pcr_tables.tex` -- spec section 4 lists exactly one
    file (not split per model), covering both the full PCR@{1,5,10,30}
    values (spec 8's "Full PCR values for every step and condition") and the
    full paired PCR deltas (spec 8's "Full paired PCR deltas for every
    step") -- the spec's required directory tree has no separate file for
    the latter, so it is a second section of this same file."""
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    checkpoints = _pcr_at_checkpoints(stepwise)
    run_inv = {(r["model"], r["dataset"], r["condition"]): r for r in _read_csv(data_dir / "run_inventory.csv")}
    delta_rows = _read_csv(data_dir / "pcr_paired_deltas_stepwise.csv")
    delta_checkpoints: dict[tuple, dict[int, dict]] = {}
    for r in delta_rows:
        step = int(r["step"])
        if step not in CHAPTER5_CHECKPOINTS:
            continue
        key = (r["model"], r["dataset"], r["comparison_name"])
        delta_checkpoints.setdefault(key, {})[step] = r

    lines = [r"% Appendix: full PCR@{1,5,10,30} values, all 12 conditions, both models (spec 8)"]
    for model in CANONICAL_BUNDLE_MODELS:
        for dataset in config.DATASETS:
            lines += [
                r"\begin{table}[t]", r"\centering",
                rf"\caption{{PCR@\{{1,5,10,30\}}, {_esc(config.DISPLAY_MODEL_NAMES.get(model, model))} / "
                rf"{_esc(config.DISPLAY_DATASET_NAMES.get(dataset, dataset))}. Denominator = pcr\_eligible runs "
                r"(canonical\_completion.py definition). Full every-step values in data/pcr_stepwise.csv.}",
                r"\begin{tabular}{lrrrrr}", r"\toprule",
                r"Condition & $n$ & PCR@1 & PCR@5 & PCR@10 & PCR@30 \\", r"\midrule",
            ]
            for cond in config.CONDITIONS:
                key = (model, dataset, cond)
                cps = checkpoints.get(key, {})
                inv = run_inv.get(key)
                if inv is None:
                    continue
                disp = _esc(config.CONDITION_DISPLAY_NAMES.get(cond, cond))
                lines.append(
                    f"{disp} & {inv['n_pcr_eligible']} & " + " & ".join(_fmt(cps.get(k), pct=True) for k in CHAPTER5_CHECKPOINTS) + r" \\"
                )
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines.append(r"% Full paired PCR deltas at PCR@{1,5,10,30} (spec 8; every-step values in data/pcr_paired_deltas_stepwise.csv)")
    for model in CANONICAL_BUNDLE_MODELS:
        for dataset in config.DATASETS:
            lines += [
                r"\begin{table}[t]", r"\centering",
                rf"\caption{{Paired PCR deltas (cue minus control), {_esc(config.DISPLAY_MODEL_NAMES.get(model, model))} / "
                rf"{_esc(config.DISPLAY_DATASET_NAMES.get(dataset, dataset))}. Positive = cue condition completed more "
                r"by step $k$ than its paired control.}",
                r"\begin{tabular}{lrrrrr}", r"\toprule",
                r"Comparison & $n_{\text{paired}}$ & $\Delta$PCR@1 & $\Delta$PCR@5 & $\Delta$PCR@10 & $\Delta$PCR@30 \\", r"\midrule",
            ]
            for comp in COMPARISONS:
                cps = delta_checkpoints.get((model, dataset, comp.comparison_name), {})
                if not cps:
                    continue
                n_paired = next(iter(cps.values())).get("n_paired", "")
                disp = _esc(comp.comparison_name)
                lines.append(
                    f"{disp} & {n_paired} & " + " & ".join(_fmt(cps.get(k, {}).get("delta_pcr"), pct=True) for k in CHAPTER5_CHECKPOINTS) + r" \\"
                )
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_full_route_tables_tex(data_dir: Path) -> str:
    rows = _read_csv(data_dir / "route_decomposition_stepwise.csv")
    last_per_cell: dict[tuple, dict] = {}
    for r in rows:
        last_per_cell[(r["model"], r["dataset"], r["condition"])] = r
    routes = ("RESEARCH_COMPLETE", "NO_TOOL_DIRECT_EXIT", "MAX_BUDGET_EXIT", "TOOL_OR_PARSER_FAILURE", "TRACE_LOGGING_FAILURE", "EXECUTION_FAILURE", "OTHER_VALID_ROUTE")
    lines = [r"% Appendix: final terminal-route tallies, all 7 canonical routes, both models (spec 8)"]
    for model in CANONICAL_BUNDLE_MODELS:
        for dataset in config.DATASETS:
            lines += [
                r"\begin{table}[t]", r"\centering",
                rf"\caption{{Final terminal-route tallies (all 7 canonical routes), "
                rf"{_esc(config.DISPLAY_MODEL_NAMES.get(model, model))} / {_esc(config.DISPLAY_DATASET_NAMES.get(dataset, dataset))}.}}",
                r"\begin{tabular}{l" + "r" * len(routes) + "}", r"\toprule",
                r"Condition & " + " & ".join(rt.replace("_", r"\_") for rt in routes) + r" \\", r"\midrule",
            ]
            for cond in config.CONDITIONS:
                r = last_per_cell.get((model, dataset, cond))
                if r is None:
                    continue
                disp = _esc(config.CONDITION_DISPLAY_NAMES.get(cond, cond))
                lines.append(disp + " & " + " & ".join(str(r.get(f"final_n_{route}", "")) for route in routes) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_evidence_tables_tex(data_dir: Path) -> str:
    rows = _read_csv(data_dir / "evidence_summary.csv")
    lines = [
        r"% Appendix: complete evidence tables, both models (spec 8)",
        r"% paired_evidence_loss = control_mean_coverage - cue_mean_coverage (spec 3.2 sign convention)",
    ]
    for model in CANONICAL_BUNDLE_MODELS:
        lines += [
            r"\begin{table}[t]", r"\centering",
            rf"\caption{{Paired evidence loss, {_esc(config.DISPLAY_MODEL_NAMES.get(model, model))}. Positive = "
            r"cue condition lost evidence relative to control.}",
            r"\begin{tabular}{llrrr}", r"\toprule",
            r"Dataset & Comparison & $n_{\text{paired}}$ & Paired loss & 95\% CI \\", r"\midrule",
        ]
        for r in rows:
            if r["model"] != model:
                continue
            lo, hi = _fmt(r.get("ci_lower"), 3), _fmt(r.get("ci_upper"), 3)
            lines.append(
                f"{_esc(config.DISPLAY_DATASET_NAMES.get(r['dataset'], r['dataset']))} & {_esc(r['comparison_name'])} & "
                f"{r.get('n_paired', '')} & {_fmt(r.get('paired_evidence_loss'), 3)} & [{lo}, {hi}] \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_answer_tables_tex(data_dir: Path) -> str:
    rows = _read_csv(data_dir / "answer_summary.csv")
    lines = [r"% Appendix: complete answer tables, both models (spec 8) -- pre-adjudication, see needs_adjudication_flag"]
    for model in CANONICAL_BUNDLE_MODELS:
        lines += [
            r"\begin{table}[t]", r"\centering",
            rf"\caption{{Final-answer accuracy (pre-adjudication), {_esc(config.DISPLAY_MODEL_NAMES.get(model, model))}.}}",
            r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Dataset & Condition & $n$ & Semantic Acc \% & EM & F1 \\", r"\midrule",
        ]
        for r in rows:
            if r["model"] != model:
                continue
            disp_cond = _esc(config.CONDITION_DISPLAY_NAMES.get(r["condition"], r["condition"]))
            lines.append(
                f"{_esc(config.DISPLAY_DATASET_NAMES.get(r['dataset'], r['dataset']))} & {disp_cond} & "
                f"{r.get('n_primary_denominator', '')} & {_fmt(r.get('pct_semantic_correct'))} & {_fmt(r.get('exact_match_mean'), 3)} & {_fmt(r.get('token_f1_mean'), 3)} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_appendix_latex_tables(bundle_root: Path) -> None:
    data_dir = bundle_root / "data"
    appendix_dir = bundle_root / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    (appendix_dir / "full_pcr_tables.tex").write_text(build_full_pcr_tables_tex(data_dir), encoding="utf-8")
    (appendix_dir / "full_route_tables.tex").write_text(build_full_route_tables_tex(data_dir), encoding="utf-8")
    (appendix_dir / "evidence_tables.tex").write_text(build_evidence_tables_tex(data_dir), encoding="utf-8")
    (appendix_dir / "answer_tables.tex").write_text(build_answer_tables_tex(data_dir), encoding="utf-8")


def write_all_tables(bundle_root: Path) -> None:
    data_dir = bundle_root / "data"
    tables_dir = bundle_root / "tables"
    write_table_index_md(tables_dir / "TABLE_INDEX.md")
    write_main_chapter_tables_md(data_dir, tables_dir / "main_chapter_tables.md")
    write_full_pcr_tables_md(data_dir, tables_dir / "full_pcr_tables.md")
    write_full_route_tables_md(data_dir, tables_dir / "full_route_tables.md")
    write_evidence_tables_md(data_dir, tables_dir / "evidence_tables.md")
    write_answer_tables_md(data_dir, tables_dir / "answer_tables.md")
    write_appendix_latex_tables(bundle_root)

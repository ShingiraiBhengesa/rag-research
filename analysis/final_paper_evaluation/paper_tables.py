"""Section 16: paper-ready LaTeX tables (main-paper compact versions).

One LaTeX file per table type per model (never pooling across models or
datasets -- each row is one dataset x condition; the file just groups them
for convenience). Booktabs style. Values pulled directly from already-computed,
already-verified CSVs -- no numbers are recomputed here.
"""

from __future__ import annotations

import csv
from pathlib import Path

from analysis import config


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt(x, nd=1, suffix=""):
    if x is None or x == "" or x == "None" or x == "NA":
        return "NA"
    try:
        return f"{float(x):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(x)


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def build_process_table_tex(checkpoint_rows: list[dict], route_summary_rows: list[dict], denom_audit_rows: list[dict], model: str) -> str:
    lines = [
        r"% Main-paper process table (Section 16.2) -- " + model,
        r"% Condition | n | PCR@1 | PCR@3 | PCR@5 | DeltaPCR@3 vs Neutral | No-additional-search % | RC % | No-tool %",
    ]
    ckpt_by_key = {(r["model"], r["dataset"], r["condition"]): r for r in checkpoint_rows}
    route_by_key = {(r["model"], r["dataset"], r["condition"]): r for r in route_summary_rows}
    denom_by_key = {(r["model"], r["dataset"], r["condition"]): r for r in denom_audit_rows}

    for dataset in config.DATASETS:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{Research-control process table, {config.DISPLAY_MODEL_NAMES.get(model, model)} / "
            rf"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}. PCR denominator = traces with an observed "
            r"terminal route in the first attacked episode (see pcr\_denominator\_audit.csv for exact n).}"
        )
        lines.append(r"\begin{tabular}{lrrrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Condition & $n$ & PCR@1 & PCR@3 & PCR@5 & $\Delta$PCR@3 vs Neutral & RC \% & No-tool \% \\")
        lines.append(r"\midrule")
        for cond in config.CONDITIONS:
            ckpt = ckpt_by_key.get((model, dataset, cond))
            route = route_by_key.get((model, dataset, cond))
            if ckpt is None or route is None:
                continue
            disp = _esc(config.CONDITION_DISPLAY_NAMES.get(cond, cond))
            lines.append(
                f"{disp} & {route['n']} & {_fmt(ckpt.get('pcr@1'))} & {_fmt(ckpt.get('pcr@3'))} & {_fmt(ckpt.get('pcr@5'))} & "
                f"{_fmt(ckpt.get('delta_pcr@3_vs_neutral'), 1)} & {_fmt(route['pct_research_complete'])} & {_fmt(route['pct_no_tool_exit'])} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    return "\n".join(lines)


def build_evidence_table_tex(evidence_cell_summary_rows: list[dict], model: str) -> str:
    lines = [
        r"% Main-paper evidence/recovery table (Section 16.3) -- " + model,
    ]
    rows_by_key = {(r["model"], r["dataset"], r["condition"]): r for r in evidence_cell_summary_rows}
    for dataset in config.DATASETS:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{Evidence coverage, {config.DISPLAY_MODEL_NAMES.get(model, model)} / "
            rf"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}. Episode-0 = union-of-retrieved coverage "
            r"at first-episode termination; final-input = citation-matched coverage of what final\_report\_generation actually received.}"
        )
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Condition & $n$ & Injection & Episode-0 & Final-input \\")
        lines.append(r"\midrule")
        for cond in config.CONDITIONS:
            row = rows_by_key.get((model, dataset, cond))
            if row is None:
                continue
            disp = _esc(config.CONDITION_DISPLAY_NAMES.get(cond, cond))
            inj = float(row["mean_coverage_at_injection"]) * 100 if row["mean_coverage_at_injection"] not in ("", "None") else None
            ep0 = float(row["mean_coverage_at_episode0_terminal"]) * 100 if row["mean_coverage_at_episode0_terminal"] not in ("", "None") else None
            fin = float(row["mean_coverage_at_final_input"]) * 100 if row["mean_coverage_at_final_input"] not in ("", "None") else None
            lines.append(f"{disp} & {row['n']} & {_fmt(inj)} & {_fmt(ep0)} & {_fmt(fin)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    return "\n".join(lines)


def build_answer_table_tex(answer_summary_rows: list[dict], model: str) -> str:
    lines = [r"% Main-paper answer table (Section 16.4) -- " + model]
    rows_by_key = {(r["model"], r["dataset"], r["condition"]): r for r in answer_summary_rows}
    for dataset in config.DATASETS:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{Final-answer accuracy, {config.DISPLAY_MODEL_NAMES.get(model, model)} / "
            rf"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}. Semantic accuracy via the validated "
            r"answer\_evaluator\_v2 procedure; gold containment is diagnostic only, not shown here.}"
        )
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Condition & Valid $n$ & Semantic Acc \% & EM & F1 \\")
        lines.append(r"\midrule")
        for cond in config.CONDITIONS:
            row = rows_by_key.get((model, dataset, cond))
            if row is None:
                continue
            disp = _esc(config.CONDITION_DISPLAY_NAMES.get(cond, cond))
            lines.append(
                f"{disp} & {row['n_primary_denominator']} & {_fmt(row['pct_semantic_correct'])} & "
                f"{_fmt(row['exact_match_mean'], 3)} & {_fmt(row['token_f1_mean'], 3)} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    return "\n".join(lines)


def build_mechanism_table_tex(mechanism_chain_rows: list[dict], model: str) -> str:
    lines = [
        r"% Mechanism table (Section 16.5) -- " + model,
        r"% Attack & Earlier completion & Unrecovered loss & Answer harm (raw counts, strict pairs only)",
    ]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Mechanism-pathway counts, {config.DISPLAY_MODEL_NAMES.get(model, model)}, strict pairs "
        r"only, vs.\ NaturalNeutral. ``Loss+harm'' requires attack completed earlier AND unrecovered "
        r"final-input evidence loss AND a control-correct/attack-wrong transition.}"
    )
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Attack & $n$ & Loss+harm & Loss, no harm \\")
    lines.append(r"\midrule")
    for row in mechanism_chain_rows:
        if row["model"] != model or row["condition_a"] != "NaturalNeutral":
            continue
        disp_ds = _esc(config.DISPLAY_DATASET_NAMES.get(row["dataset"], row["dataset"]))
        disp_cond = _esc(config.CONDITION_DISPLAY_NAMES.get(row["condition_b"], row["condition_b"]))
        lines.append(
            f"{disp_ds} & {disp_cond} & {row['n']} & "
            f"{row['earlier_completion_and_unrecovered_loss_and_harm']} & {row['earlier_completion_and_unrecovered_loss_and_no_harm']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_all_paper_tables(new_output_root: Path, old_pcr_root: Path) -> None:
    out_dir = new_output_root / "paper_ready"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = _read_csv(old_pcr_root / "pcr" / "pcr_checkpoint_effects.csv")
    route_summary_rows = _read_csv(old_pcr_root / "pcr" / "pcr_terminal_route_summary.csv")
    denom_audit_rows = _read_csv(old_pcr_root / "pcr" / "pcr_denominator_audit.csv")
    evidence_cell_summary_rows = _read_csv(old_pcr_root / "evidence" / "evidence_coverage_cell_summary.csv")
    answer_summary_rows = _read_csv(old_pcr_root / "answers" / "final_answer_summary.csv")
    mechanism_chain_rows = _read_csv(new_output_root / "tables" / "mechanism_chain_summary.csv")

    for model in config.INCLUDED_MODELS:
        (out_dir / f"process_table_{model}.tex").write_text(
            build_process_table_tex(checkpoint_rows, route_summary_rows, denom_audit_rows, model), encoding="utf-8"
        )
        (out_dir / f"evidence_table_{model}.tex").write_text(
            build_evidence_table_tex(evidence_cell_summary_rows, model), encoding="utf-8"
        )
        (out_dir / f"answer_table_{model}.tex").write_text(
            build_answer_table_tex(answer_summary_rows, model), encoding="utf-8"
        )
        (out_dir / f"mechanism_table_{model}.tex").write_text(
            build_mechanism_table_tex(mechanism_chain_rows, model), encoding="utf-8"
        )

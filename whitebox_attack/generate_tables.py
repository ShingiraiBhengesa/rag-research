"""
generate_tables.py — Generate result tables from PCR@1–10 JSONL outputs.

Reads:
  ../whitebox_results/s1_attack2e_pcr10_{qwen7b,qwen14b,llama8b}.jsonl

Saves to ../whitebox_results/tables/:
  pcr_per_dataset.tex   — per-dataset LaTeX tables (PCR@1/3/5/10 + Net)
  pcr_per_dataset.txt   — plain-text equivalent
  pcr_summary.tex       — cross-dataset summary table (Net@3 and Net@5)
  pcr_summary.txt       — plain-text equivalent

Run:
  cd whitebox_attack/
  python generate_tables.py
"""

import os
import json
import collections

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "whitebox_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_SOURCES = {
    "Qwen2.5-7B":   os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_qwen7b.jsonl"),
    "Qwen2.5-14B":  os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_qwen14b.jsonl"),
    "Llama-3.1-8B": os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_llama8b.jsonl"),
}

DATASET_MAP = {
    "hotpot":      "HotpotQA",
    "2wiki":       "2WikiMultiHopQA",
    "musique":     "MuSiQue",
    "multihoprag": "MultiHopRAG",
}

MODELS   = list(JSONL_SOURCES.keys())
DATASETS = list(DATASET_MAP.values())
STEPS    = list(range(1, 11))

SHOW_K   = list(range(1, 11))   # all steps 1–10

COND_KEYS = {
    "clean":                  "Clean",
    "epistemic_confirmation": "Epist.",
    "search_match_cue":       "S-Match",
}

COND_ORDER = ["clean", "epistemic_confirmation", "search_match_cue"]


def _pcr_field(k):
    return "wrong_gap_assessment_success" if k == 1 else f"delayed_pcr_at_{k}"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    """Return data[dataset][model][cond_key] = {k: pct, ...} and n[dataset][model][cond_key]."""
    acc = collections.defaultdict(list)
    for model_label, path in JSONL_SOURCES.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        with open(path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not row.get("eligible"):
                    continue
                ck = row.get("condition")
                if ck not in COND_KEYS:
                    continue
                ds = DATASET_MAP.get(row.get("dataset", ""))
                if ds is None:
                    continue
                for k in STEPS:
                    val = row.get(_pcr_field(k))
                    if val is not None:
                        acc[(ds, model_label, ck, k)].append(bool(val))

    rates, counts = {}, {}
    for ds in DATASETS:
        rates[ds], counts[ds] = {}, {}
        for model in MODELS:
            rates[ds][model], counts[ds][model] = {}, {}
            for ck in COND_KEYS:
                rates[ds][model][ck]  = {}
                counts[ds][model][ck] = len(acc[(ds, model, ck, 1)])
                for k in STEPS:
                    lst = acc[(ds, model, ck, k)]
                    rates[ds][model][ck][k] = 100.0 * sum(lst) / len(lst) if lst else 0.0
    return rates, counts


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _pct(v):
    return f"{v:.1f}"


def _net(attack, clean):
    delta = attack - clean
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


# ── Per-dataset plain-text tables ──────────────────────────────────────────────
def make_text_per_dataset(rates, counts):
    lines = []
    step_header = "  ".join(f"@{k:2d}" for k in SHOW_K)
    sep_cols    = "  ".join(["─" * 5] * len(SHOW_K))
    width = 16 + 2 + 10 + 2 + len(step_header) + 8

    for ds in DATASETS:
        lines.append(f"\n{'─' * width}")
        lines.append(f"  {ds}")
        lines.append(f"{'─' * width}")
        lines.append(
            f"  {'Model':<16}  {'Condition':<10}  " + step_header
        )
        lines.append(f"  {'─'*16}  {'─'*10}  {sep_cols}")

        for model in MODELS:
            lines.append("")
            for i, ck in enumerate(COND_ORDER):
                label     = COND_KEYS[ck]
                r         = rates[ds][model][ck]
                n         = counts[ds][model][ck]
                model_str = model if i == 0 else ""
                vals      = "  ".join(f"{_pct(r[k]):>5}" for k in SHOW_K)
                n_str     = f"(n={n})" if i == 0 else ""
                lines.append(f"  {model_str:<16}  {label:<10}  {vals}  {n_str}")

    lines.append(f"\n{'─' * width}")
    return "\n".join(lines)


# ── Per-dataset LaTeX tables ───────────────────────────────────────────────────
LATEX_MODEL_DISPLAY = {
    "Qwen2.5-7B":   r"Qwen2.5-7B",
    "Qwen2.5-14B":  r"Qwen2.5-14B",
    "Llama-3.1-8B": r"Llama-3.1-8B",
}

def make_latex_per_dataset(rates, counts):
    # 2 fixed cols (model, condition) + 10 PCR cols
    col_spec = "ll" + "r" * len(SHOW_K)

    lines = []
    lines.append(r"% ── Per-dataset PCR@1–10 tables ─────────────────────────────────────────────")
    lines.append(r"% Auto-generated by generate_tables.py")
    lines.append(r"% Columns: Model | Condition | PCR@1 … PCR@10")
    lines.append("")

    step_header = " & ".join(f"\\textbf{{@{k}}}" for k in SHOW_K)

    for ds in DATASETS:
        ds_label = ds
        ds_key   = ds.lower().replace(" ", "_").replace("multihopqa", "mhqa")
        lines.append(f"% ── {ds} ──")
        lines.append(r"\begin{table}[t]")
        lines.append(r"  \centering")
        lines.append(r"  \footnotesize")
        lines.append(r"  \setlength{\tabcolsep}{4pt}")
        lines.append(f"  \\caption{{Premature Completion Rate PCR@k (\\%) on {ds_label}. "
                     r"Steps 1–10; clean baseline vs.\ two attack cues.}}")
        lines.append(f"  \\label{{tab:pcr_{ds_key}}}")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")
        lines.append(
            f"    \\textbf{{Model}} & \\textbf{{Condition}} & {step_header} \\\\"
        )
        lines.append(r"    \midrule")

        for mi, model in enumerate(MODELS):
            if mi > 0:
                lines.append(r"    \midrule")
            n_clean = counts[ds][model]["clean"]

            for i, ck in enumerate(COND_ORDER):
                label  = COND_KEYS[ck]
                r      = rates[ds][model][ck]
                n      = counts[ds][model][ck]
                mname  = f"\\textbf{{{LATEX_MODEL_DISPLAY[model]}}}" if i == 0 else ""
                n_note = f"\\,({n})" if i == 0 else ""
                vals   = " & ".join(_pct(r[k]) for k in SHOW_K)
                lines.append(f"    {mname}{n_note} & {label} & {vals} \\\\")

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    return "\n".join(lines)


# ── Cross-dataset summary: all steps ──────────────────────────────────────────
def make_text_summary(rates):
    """One block per dataset×condition, columns = steps 1–10."""
    lines = []
    step_header = "  ".join(f"@{k:2d}" for k in SHOW_K)
    sep_cols    = "  ".join(["─" * 5] * len(SHOW_K))
    width = 16 + 2 + 10 + 2 + len(step_header) + 8

    lines.append(f"\n{'─' * width}")
    lines.append("  Summary: PCR@1–10 (%) across all datasets")
    lines.append(f"{'─' * width}")

    for ds in DATASETS:
        lines.append(f"\n  ── {ds} ──")
        lines.append(f"  {'Model':<16}  {'Condition':<10}  {step_header}")
        lines.append(f"  {'─'*16}  {'─'*10}  {sep_cols}")
        for model in MODELS:
            lines.append("")
            for i, ck in enumerate(COND_ORDER):
                label     = COND_KEYS[ck]
                r         = rates[ds][model][ck]
                n         = rates[ds][model][ck]   # reuse dict key
                model_str = model if i == 0 else ""
                vals      = "  ".join(f"{_pct(rates[ds][model][ck][k]):>5}" for k in SHOW_K)
                lines.append(f"  {model_str:<16}  {label:<10}  {vals}")

    lines.append(f"\n{'─' * width}")
    return "\n".join(lines)


def make_latex_summary(rates, counts):
    """Cross-dataset summary: one sub-table per dataset, rows = model×condition, cols = @1–@10."""
    col_spec = "ll" + "r" * len(SHOW_K)
    step_header = " & ".join(f"\\textbf{{@{k}}}" for k in SHOW_K)

    lines = []
    lines.append(r"% ── Cross-dataset PCR@1–10 summary ─────────────────────────────────────────")
    lines.append(r"% Auto-generated by generate_tables.py")
    lines.append(r"% Four tables (one per dataset), same format as per-dataset tables.")
    lines.append("")

    for ds in DATASETS:
        ds_key = ds.lower().replace(" ", "_").replace("multihopqa", "mhqa")
        lines.append(f"% ── {ds} (summary) ──")
        lines.append(r"\begin{table}[t]")
        lines.append(r"  \centering")
        lines.append(r"  \footnotesize")
        lines.append(r"  \setlength{\tabcolsep}{4pt}")
        lines.append(f"  \\caption{{PCR@k (\\%) on {ds}. Steps 1–10; "
                     r"clean baseline and two attack cues.}}")
        lines.append(f"  \\label{{tab:summary_{ds_key}}}")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")
        lines.append(f"    \\textbf{{Model}} & \\textbf{{Condition}} & {step_header} \\\\")
        lines.append(r"    \midrule")

        for mi, model in enumerate(MODELS):
            if mi > 0:
                lines.append(r"    \midrule")
            n_clean = counts[ds][model]["clean"]
            for i, ck in enumerate(COND_ORDER):
                label  = COND_KEYS[ck]
                r      = rates[ds][model][ck]
                n      = counts[ds][model][ck]
                mname  = f"\\textbf{{{LATEX_MODEL_DISPLAY[model]}}}" if i == 0 else ""
                n_note = f"\\,({n})" if i == 0 else ""
                vals   = " & ".join(_pct(r[k]) for k in SHOW_K)
                lines.append(f"    {mname}{n_note} & {label} & {vals} \\\\")

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data …")
    rates, counts = load_data()

    # Per-dataset tables
    txt_per_ds = make_text_per_dataset(rates, counts)
    tex_per_ds = make_latex_per_dataset(rates, counts)

    # Summary table
    txt_summary = make_text_summary(rates)
    tex_summary = make_latex_summary(rates, counts)

    # Write files
    for fname, content in [
        ("pcr_per_dataset.txt", txt_per_ds),
        ("pcr_per_dataset.tex", tex_per_ds),
        ("pcr_summary.txt",     txt_summary),
        ("pcr_summary.tex",     tex_summary),
    ]:
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            f.write(content + "\n")
        print(f"  saved: {path}")

    # Also print plain-text to stdout
    print(txt_per_ds)
    print(txt_summary)

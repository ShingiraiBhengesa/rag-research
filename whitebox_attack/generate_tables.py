"""
generate_tables.py — Generate PCR@k result tables from final Wrong Gap Assessment JSONL.

Reads:
  ../whitebox_results/final_wga_{qwen7b,qwen14b,llama8b,gemma27b,gptoss20b}.jsonl

Derives PCR@k from completion_step_* fields:
  pcr_at_k = completion_step is not None and completion_step <= k

Saves to ../whitebox_results/tables/:
  wga_per_dataset.tex / .txt  — per-dataset tables (all 5 conditions × selected k)
  wga_summary.tex / .txt      — cross-dataset summary

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
    "Qwen2.5-7B":   os.path.join(RESULTS_DIR, "final_wga_qwen7b.jsonl"),
    "Qwen2.5-14B":  os.path.join(RESULTS_DIR, "final_wga_qwen14b.jsonl"),
    "Llama-3.1-8B": os.path.join(RESULTS_DIR, "final_wga_llama8b.jsonl"),
    "Gemma-3-27B":  os.path.join(RESULTS_DIR, "final_wga_gemma27b.jsonl"),
    "GPT-OSS-20B":  os.path.join(RESULTS_DIR, "final_wga_gptoss20b.jsonl"),
}

DATASET_MAP = {
    "hotpot":      "HotpotQA",
    "2wiki":       "2WikiMultiHopQA",
    "musique":     "MuSiQue",
    "multihoprag": "MultiHopRAG",
}

MODELS   = list(JSONL_SOURCES.keys())
DATASETS = list(DATASET_MAP.values())

# Conditions in display order
COND_ORDER = [
    "clean",
    "epistemic_confirmation",
    "search_match_cue",
    "source_coverage",
    "epistemic_confirmation_source",
    "search_match_source",
]
COND_LABEL = {
    "clean":                         "Clean",
    "epistemic_confirmation":        "Epist.",
    "search_match_cue":              "S-Match",
    "source_coverage":               "Src-Cov",
    "epistemic_confirmation_source": "Epist.+Src",
    "search_match_source":           "SMatch+Src",
}

# Steps shown in tables (selected; dynamic max used in figures)
SHOW_K = [1, 2, 3, 5, 10, 20]

LATEX_MODEL = {
    "Qwen2.5-7B":   r"Qwen2.5-7B",
    "Qwen2.5-14B":  r"Qwen2.5-14B",
    "Llama-3.1-8B": r"Llama-3.1-8B",
    "Gemma-3-27B":  r"Gemma-3-27B",
    "GPT-OSS-20B":  r"GPT-OSS-20B",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    """Return (rates, counts, max_step).

    rates[ds][model][cond][k]  = PCR@k as a percentage (float)
    counts[ds][model][cond]    = n eligible rows
    max_step                   = largest finite completion step observed
    """
    # acc[(ds_label, model_label, cond)] = list of completion_step (int or None)
    acc      = collections.defaultdict(list)
    max_step = 10

    for model_label, path in JSONL_SOURCES.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — {model_label} skipped")
            continue
        with open(path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not row.get("eligible"):
                    continue
                cond = row.get("condition")
                if cond not in COND_ORDER:
                    continue
                ds = DATASET_MAP.get(row.get("dataset", ""))
                if ds is None:
                    continue
                cs = (row.get("completion_step_clean")
                      if cond == "clean"
                      else row.get("completion_step_attack"))
                acc[(ds, model_label, cond)].append(cs)
                if cs is not None and cs > max_step:
                    max_step = cs

    def pcr_at(comp_steps, k):
        n = len(comp_steps)
        if n == 0:
            return 0.0
        return 100.0 * sum(1 for s in comp_steps if s is not None and s <= k) / n

    rates, counts = {}, {}
    for ds in DATASETS:
        rates[ds], counts[ds] = {}, {}
        for model in MODELS:
            rates[ds][model], counts[ds][model] = {}, {}
            for cond in COND_ORDER:
                comp = acc[(ds, model, cond)]
                counts[ds][model][cond] = len(comp)
                max_k = max(SHOW_K + [max_step])
                rates[ds][model][cond] = {k: pcr_at(comp, k) for k in range(1, max_k + 1)}

    return rates, counts, max_step


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _pct(v):
    return f"{v:.1f}"

def _net(attack_v, clean_v):
    d = attack_v - clean_v
    return f"{'+'if d>=0 else ''}{d:.1f}"


# ── Plain-text table ───────────────────────────────────────────────────────────

def make_text_table(rates, counts, max_step, label="Per-dataset"):
    ks       = [k for k in SHOW_K if k <= max(SHOW_K[:-1] + [max_step])]
    hdr_ks   = "  ".join(f"@ {k:2d}" for k in ks)
    sep_ks   = "  ".join(["─" * 5] * len(ks))
    W        = 16 + 2 + 12 + 2 + len(hdr_ks) + 4

    lines = [f"\n{'─'*W}", f"  {label}: Wrong Gap Assessment PCR@k (%)", f"{'─'*W}"]

    for ds in DATASETS:
        lines += [f"\n  ── {ds} ──",
                  f"  {'Model':<16}  {'Condition':<12}  {hdr_ks}",
                  f"  {'─'*16}  {'─'*12}  {sep_ks}"]
        for model in MODELS:
            lines.append("")
            clean_r = rates[ds][model]["clean"]
            for i, cond in enumerate(COND_ORDER):
                label_s   = COND_LABEL[cond]
                r         = rates[ds][model][cond]
                n         = counts[ds][model][cond]
                model_s   = model if i == 0 else ""
                vals      = "  ".join(f"{_pct(r[k]):>5}" for k in ks)
                net_s     = "" if cond == "clean" else \
                            f"  (Net@3={_net(r[3], clean_r[3])}pp, @10={_net(r[10], clean_r[10])}pp)"
                n_s       = f"  n={n}" if i == 0 else ""
                lines.append(f"  {model_s:<16}  {label_s:<12}  {vals}{net_s}{n_s}")

    lines.append(f"\n{'─'*W}")
    return "\n".join(lines)


# ── LaTeX tables ───────────────────────────────────────────────────────────────

def make_latex_table(rates, counts, max_step):
    ks         = [k for k in SHOW_K if k <= max(SHOW_K[:-1] + [max_step])]
    col_spec   = "ll" + "r" * len(ks)
    step_hdr   = " & ".join(f"\\textbf{{@{k}}}" for k in ks)

    lines = [
        r"% Wrong Gap Assessment PCR@k tables — auto-generated by generate_tables.py",
        r"% Five conditions per model: epistemic_confirmation, search_match_cue,",
        r"% source_coverage, epistemic_confirmation_source, search_match_source",
        "",
    ]

    for ds in DATASETS:
        ds_key = {"HotpotQA": "hotpot", "2WikiMultiHopQA": "2wiki",
                  "MuSiQue": "musique", "MultiHopRAG": "multihoprag"}[ds]
        lines += [
            f"% ── {ds} ──",
            r"\begin{table}[t]",
            r"  \centering\footnotesize\setlength{\tabcolsep}{4pt}",
            f"  \\caption{{Wrong Gap Assessment PCR@k (\\%) on {ds}. "
            r"Five attack cues vs.\ clean baseline; all five models.}",
            f"  \\label{{tab:wga_{ds_key}}}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            f"    \\textbf{{Model}} & \\textbf{{Condition}} & {step_hdr} \\\\",
            r"    \midrule",
        ]

        for mi, model in enumerate(MODELS):
            if mi > 0:
                lines.append(r"    \midrule")
            clean_r = rates[ds][model]["clean"]
            n_clean = counts[ds][model]["clean"]
            for i, cond in enumerate(COND_ORDER):
                label_s = COND_LABEL[cond]
                r       = rates[ds][model][cond]
                n       = counts[ds][model][cond]
                mname   = f"\\textbf{{{LATEX_MODEL[model]}}}" if i == 0 else ""
                n_note  = f"\\,({n_clean})" if i == 0 else ""
                vals    = " & ".join(_pct(r[k]) for k in ks)
                lines.append(f"    {mname}{n_note} & {label_s} & {vals} \\\\")

        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data …")
    rates, counts, max_step = load_data()
    print(f"Max completion step observed: {max_step}")

    txt = make_text_table(rates, counts, max_step)
    tex = make_latex_table(rates, counts, max_step)

    for fname, content in [
        ("wga_per_dataset.txt", txt),
        ("wga_per_dataset.tex", tex),
    ]:
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            f.write(content + "\n")
        print(f"  saved: {path}")

    print(txt)

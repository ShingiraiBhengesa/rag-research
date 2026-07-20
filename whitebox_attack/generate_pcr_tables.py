"""
generate_pcr_tables.py — Regenerate PCR@k tables with correct intersection-aligned baselines.

Reads per model:
  final_wga_{mk}.jsonl          — WGA full attack + clean
  final_wga_factonly_{mk}.jsonl — FactOnly (v2 preferred for GPT-OSS if it exists)
  final_wga_cueonly_{mk}.jsonl  — CueOnly ablation

Baseline alignment
------------------
The eligible set per (model, dataset) is the INTERSECTION of WGA-eligible and FO-eligible
example_ids. This ensures all conditions share the same denominator regardless of
FlashAttention non-determinism between independent clean rollouts.

For GPT-OSS: if final_wga_factonly_gptoss20b_v2.jsonl exists, it is used in preference
to the old _v1 file. The v2 file is manifest-driven (inherits WGA eligibility), so its
intersection with WGA is the full WGA eligible set.

CueOnly is already manifest-driven (inherits WGA eligibility), so it naturally falls
within the WGA eligible set; no intersection adjustment needed beyond the WGA/FO alignment.

Outputs (written to whitebox_results/tables/):
  pcr_tables.md   — Markdown tables, k=1–15, all 12 conditions per model/dataset
  pcr_tables.html — Standalone HTML version

Run:
  cd whitebox_attack/
  python generate_pcr_tables.py
"""

import collections
import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "whitebox_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("Qwen2.5-7B",   "qwen7b"),
    ("Qwen2.5-14B",  "qwen14b"),
    ("Llama-3.1-8B", "llama8b"),
    ("Gemma-3-27B",  "gemma27b"),
    ("GPT-OSS-20B",  "gptoss20b"),
]

DATASETS = ["hotpot", "2wiki", "musique", "multihoprag"]
DATASET_LABEL = {
    "hotpot":      "HotpotQA",
    "2wiki":       "2WikiMultiHopQA",
    "musique":     "MuSiQue",
    "multihoprag": "MultiHopRAG",
}

WGA_FAMILIES = [
    "epistemic_confirmation",
    "search_match_cue",
    "source_coverage",
    "epistemic_confirmation_source",
    "search_match_source",
]
CO_CONDITIONS = [f"cue_only_{f}" for f in WGA_FAMILIES]

COND_LABEL = {
    "clean":                                    "Clean",
    "factonly":                                 "FactOnly",
    "cue_only_epistemic_confirmation":          "CO: Epist.Confirm",
    "cue_only_search_match_cue":                "CO: SearchMatch",
    "cue_only_source_coverage":                 "CO: SrcCoverage",
    "cue_only_epistemic_confirmation_source":   "CO: Epist+Src",
    "cue_only_search_match_source":             "CO: Match+Src",
    "epistemic_confirmation":                   "WGA: Epist.Confirm",
    "search_match_cue":                         "WGA: SearchMatch",
    "source_coverage":                          "WGA: SrcCoverage",
    "epistemic_confirmation_source":            "WGA: Epist+Src",
    "search_match_source":                      "WGA: Match+Src",
}

DISPLAY_ORDER = [
    "clean",
    "factonly",
    *CO_CONDITIONS,
    *WGA_FAMILIES,
]

MAX_K = 15


# ── Data loading ───────────────────────────────────────────────────────────────

def load_eligible_ids(path, condition_filter="clean"):
    """Return {dataset: set(example_id)} for rows where eligible=True and condition matches."""
    result = collections.defaultdict(set)
    if not os.path.exists(path):
        return result
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if row.get("condition") != condition_filter:
                continue
            if not row.get("eligible"):
                continue
            ds = row.get("dataset", "")
            eid = row.get("example_id", "")
            if ds and eid:
                result[ds].add(eid)
    return result


def load_completion_steps(path, eligible_ids_per_ds, conditions):
    """Return {condition: {dataset: [completion_step or None]}} for eligible_ids_per_ds.

    eligible_ids_per_ds: {dataset: set(example_id)} — intersection-aligned eligible set.
    conditions: iterable of condition strings to collect.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    if not os.path.exists(path):
        return acc
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not row.get("eligible"):
                continue
            cond = row.get("condition", "")
            if cond not in conditions:
                continue
            ds = row.get("dataset", "")
            eid = row.get("example_id", "")
            if ds not in eligible_ids_per_ds or eid not in eligible_ids_per_ds[ds]:
                continue
            # completion_step field depends on condition type
            if cond == "clean":
                cs = row.get("completion_step_clean")
            else:
                cs = row.get("completion_step_attack")
            acc[cond][ds].append(cs)
    return acc


def pcr_at(comp_steps, k):
    n = len(comp_steps)
    if n == 0:
        return None
    return 100.0 * sum(1 for s in comp_steps if s is not None and s <= k) / n


# ── Main data collection ───────────────────────────────────────────────────────

def collect_all():
    """
    Returns data[(model_label, dataset, condition)] = list[completion_step or None]
    using intersection-aligned eligible sets.
    """
    data = {}

    for model_label, mk in MODELS:
        wga_path = os.path.join(RESULTS_DIR, f"final_wga_{mk}.jsonl")
        # Prefer v2 FO for GPT-OSS if it exists (manifest-driven, zero drift)
        if mk == "gptoss20b":
            fo_path_v2 = os.path.join(RESULTS_DIR, "final_wga_factonly_gptoss20b_v2.jsonl")
            fo_path    = fo_path_v2 if os.path.exists(fo_path_v2) else \
                         os.path.join(RESULTS_DIR, f"final_wga_factonly_{mk}.jsonl")
        else:
            fo_path = os.path.join(RESULTS_DIR, f"final_wga_factonly_{mk}.jsonl")
        co_path = os.path.join(RESULTS_DIR, f"final_wga_cueonly_{mk}.jsonl")

        print(f"\n  {model_label}:")
        print(f"    WGA  : {os.path.basename(wga_path)}  exists={os.path.exists(wga_path)}")
        print(f"    FO   : {os.path.basename(fo_path)}  exists={os.path.exists(fo_path)}")
        print(f"    CO   : {os.path.basename(co_path)}  exists={os.path.exists(co_path)}")

        # Build intersection of WGA-eligible and FO-eligible per dataset
        wga_elig = load_eligible_ids(wga_path, "clean")
        fo_elig  = load_eligible_ids(fo_path,  "clean")

        inter = {}
        for ds in DATASETS:
            w = wga_elig.get(ds, set())
            # For manifest-driven FO (v2 or Llama8B/Gemma27B perfect match),
            # FO eligible = WGA eligible, so intersection = WGA eligible.
            # For others with small drift, intersection removes 0-6 examples.
            f = fo_elig.get(ds, set())
            # If FO file is missing or empty, fall back to WGA eligible only
            inter[ds] = (w & f) if f else w
            print(f"    {ds:14s}: WGA={len(w):3d}  FO={len(f):3d}  "
                  f"intersection={len(inter[ds]):3d}", flush=True)

        # Load completion steps for all conditions, restricted to intersection
        wga_steps = load_completion_steps(
            wga_path, inter, {"clean"} | set(WGA_FAMILIES))
        fo_steps  = load_completion_steps(
            fo_path,  inter, {"factonly"})
        co_steps  = load_completion_steps(
            co_path,  inter, set(CO_CONDITIONS))

        for ds in DATASETS:
            for cond in DISPLAY_ORDER:
                if cond in wga_steps:
                    steps = wga_steps[cond].get(ds, [])
                elif cond in fo_steps:
                    steps = fo_steps[cond].get(ds, [])
                elif cond in co_steps:
                    steps = co_steps[cond].get(ds, [])
                else:
                    steps = []
                data[(model_label, ds, cond)] = steps

    return data


# ── Markdown output ────────────────────────────────────────────────────────────

def make_markdown(data):
    ks = list(range(1, MAX_K + 1))

    def row_md(label, n, steps_or_none):
        if steps_or_none is None or len(steps_or_none) == 0:
            vals = " | ".join(["   —  "] * len(ks))
            return f"| {label:<26}| {'—':>6} | {vals} |"
        vals = " | ".join(
            f"{pcr_at(steps_or_none, k):>5.1f}" for k in ks)
        n_str = str(n)
        return f"| {label:<26}| {n_str:>6} | {vals} |"

    hdr_ks = " | ".join(f"@{k:>2}" for k in ks)
    lines  = [
        "# PCR@k — All Attack Families, k=1–15",
        "Eligible rows only. Intersection-aligned baseline: eligible in both WGA and FO. "
        "Dashes = data not yet collected.",
    ]

    for model_label, _ in MODELS:
        lines.append(f"\n## {model_label}\n")
        for ds in DATASETS:
            lines.append(f"### {DATASET_LABEL[ds]}\n")
            lines.append(
                f"| {'Condition':<26}| {'n':>6} | {hdr_ks} |"
            )
            sep_vals = " | ".join(["------"] * len(ks))
            lines.append(
                f"| {'-'*26}|{'-'*7}| {sep_vals} |"
            )

            sep_added = {"factonly": False, "cue_only_epistemic_confirmation": False,
                         "epistemic_confirmation": False}

            for cond in DISPLAY_ORDER:
                label = COND_LABEL.get(cond, cond)
                # Add section dividers
                if cond == "cue_only_epistemic_confirmation":
                    lines.append(f"| **{'Cue Only (D1 + suffix)':<24}**|       |{' | '.join(['      '] * len(ks))} |")
                elif cond == "epistemic_confirmation":
                    lines.append(f"| **{'WGA Full Attack (D1+F2+sfx)':<24}**|       |{' | '.join(['      '] * len(ks))} |")

                steps = data.get((model_label, ds, cond), [])
                lines.append(row_md(label, len(steps), steps if steps else None))
            lines.append("")

    return "\n".join(lines)


# ── HTML output ────────────────────────────────────────────────────────────────

def make_html(data):
    ks = list(range(1, MAX_K + 1))

    def _pct(v):
        return f"{v:.1f}"

    rows_html = []
    for model_label, _ in MODELS:
        for ds in DATASETS:
            for i, cond in enumerate(DISPLAY_ORDER):
                steps = data.get((model_label, ds, cond), [])
                n     = len(steps)
                is_section_header = cond in (
                    "cue_only_epistemic_confirmation",
                    "epistemic_confirmation",
                )
                if is_section_header:
                    section_name = ("Cue Only" if "cue" in cond else "WGA Full")
                    rows_html.append(
                        f'<tr class="section-header">'
                        f'<td>{model_label}</td><td>{DATASET_LABEL[ds]}</td>'
                        f'<td colspan="{len(ks)+2}"><em>{section_name}</em></td></tr>'
                    )
                cls  = "clean" if cond == "clean" else ""
                vals = "".join(
                    f"<td>{_pct(pcr_at(steps, k)) if steps else '—'}</td>"
                    for k in ks
                )
                label = COND_LABEL.get(cond, cond)
                rows_html.append(
                    f'<tr class="{cls}">'
                    f"<td>{model_label if i==0 else ''}</td>"
                    f"<td>{DATASET_LABEL[ds] if i==0 else ''}</td>"
                    f"<td>{label}</td><td>{n if steps else '—'}</td>"
                    f"{vals}</tr>"
                )

    k_headers = "".join(f"<th>@{k}</th>" for k in ks)
    table_html = (
        f"<thead><tr><th>Model</th><th>Dataset</th><th>Condition</th>"
        f"<th>n</th>{k_headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCR@k Tables — Wrong Gap Assessment</title>
<style>
  body {{ font-family: system-ui, sans-serif; font-size: 13px; padding: 1rem 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
  tr.clean td {{ background: #f9f9f9; font-weight: bold; }}
  tr.section-header td {{ background: #e8e8f0; font-style: italic; text-align: left; }}
  tr:hover td {{ background: #fff8dc; }}
</style>
</head>
<body>
<h1>PCR@k — All Attack Families, k=1–15</h1>
<p>Eligible rows only. Intersection-aligned baseline (eligible in both WGA and FO).
Dashes = data not yet collected.</p>
<div style="overflow-x:auto">
<table>{table_html}</table>
</div>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Collecting PCR data with intersection-aligned eligible sets ===")
    data = collect_all()

    print("\n=== Writing tables ===")
    md   = make_markdown(data)
    html = make_html(data)

    md_path   = os.path.join(OUT_DIR, "pcr_tables.md")
    html_path = os.path.join(OUT_DIR, "pcr_tables.html")

    with open(md_path,   "w") as f:
        f.write(md + "\n")
    with open(html_path, "w") as f:
        f.write(html + "\n")

    print(f"  saved: {md_path}")
    print(f"  saved: {html_path}")

    # Quick completeness summary
    print("\n=== Completeness summary (n per model/dataset/condition) ===")
    for model_label, _ in MODELS:
        print(f"\n  {model_label}:")
        for ds in DATASETS:
            clean_n = len(data.get((model_label, ds, "clean"), []))
            fo_n    = len(data.get((model_label, ds, "factonly"), []))
            co_n    = len(data.get((model_label, ds, "cue_only_epistemic_confirmation"), []))
            wga_n   = len(data.get((model_label, ds, "epistemic_confirmation"), []))
            print(f"    {ds:14s}: clean={clean_n:3d}  fo={fo_n:3d}  "
                  f"co_epist={co_n:3d}  wga_epist={wga_n:3d}")

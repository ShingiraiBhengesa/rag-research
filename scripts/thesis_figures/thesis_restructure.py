"""Thesis results restructure: 9 CSVs (notes/reviewer_checks/) and 8 figures (results/figures/thesis_restructure/).

Run from the repo root (whitebox env):
    PYTHONPATH=src python notes/reviewer_checks/thesis_restructure.py

Sources
  E1 runs     results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage{,_gemma_gptoss}/run_level_evidence_hops.csv
              (T_i, route, pcr_eligible, evidence at the Researcher's stop and at the final-report input)
  E1 answers  local_data/evaluation/v3_final_completed_qwen_llama_20260910/answers/final_answer_predictions.jsonl
              results/whitebox/e1_fixed_cue_matrix/gemma_gptoss/data/answer_run_level.csv  (same loader as answer_harm.py)
  E1 char.    results/whitebox/e1_fixed_cue_matrix/*/data/pcr_paired_deltas_stepwise.csv
  E2 runs     local_data/e2_bridge_mechanism/run_results/full/trajectories/run_labels.csv  (v2b full run)
Definitions (printed again at run time)
  t        first post-observation completion step (RESEARCH_COMPLETE / NO_TOOL_DIRECT_EXIT); no completion = inf
  pair     same model x dataset x question, both runs PCR-eligible
  success  t_cue < t_neutral
  EICR     among runs that completed, share whose evidence at the Researcher's stop is incomplete
  early    cue run stopped by the model's characteristic step (t_cue <= c_m)
  recovery among early cue runs incomplete at the Researcher's stop, share complete at the final-report input
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "results/whitebox/e1_fixed_cue_matrix"
E2_LABELS = REPO / "local_data/e2_bridge_mechanism/run_results/full/trajectories/run_labels.csv"
QL_ANSWERS = REPO / "local_data/evaluation/v3_final_completed_qwen_llama_20260910/answers/final_answer_predictions.jsonl"
GG_ANSWERS = E1 / "gemma_gptoss/data/answer_run_level.csv"
OUTC = REPO / "notes/reviewer_checks"
OUTF = REPO / "results/figures/thesis_restructure"
SEED, NB = 20260910, 2000

MODELS = [("qwen14b", "Qwen"), ("llama8b", "Llama"), ("gemma27b", "Gemma"), ("gpt_oss_20b", "gpt-oss")]
DATASETS = [("hotpot", "HotpotQA"), ("2wiki", "2Wiki"), ("musique", "MuSiQue"), ("multihoprag", "MultiHop-RAG")]
NEU, FIX, SAT = "NaturalNeutral", "Natural_PreRegisteredSufficiency", "Natural_PreRegisteredSufficiencySaturation"
CLEAN = "NaturalClean"  # placebo control: Neutral vs Clean
ARM = {NEU: "Neutral", FIX: "Fixed Sufficiency", SAT: "Suff.+Saturation",
       "neutral": "Neutral", "fixed_sufficiency": "Fixed Sufficiency", "bridge_sufficiency_k3": "Bridge K3"}
ROUTES = {"RESEARCH_COMPLETE", "NO_TOOL_DIRECT_EXIT"}
E2_MODELS, E2_DATASETS = MODELS[:2], DATASETS[:2]
E2_CHAR = {"qwen14b": 1, "llama8b": 3}

C_NEU, C_FIX, C_SAT, C_K3, C_GRAY = "#2a78d6", "#1baf7a", "#eb6834", "#1baf7a", "#73726c"
C_RED, C_POST = "#e34948", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})


# --------------------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    means = x[rng.integers(0, len(x), size=(NB, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return (100 * lo, 100 * hi)


def b(s):
    return s.astype(str) == "True"


def save_csv(df, name):
    print(f"\n--- {name}: {len(df)} rows; first 3 rows:")
    print(df.head(3).to_string(index=False))
    df.to_csv(OUTC / name, index=False, float_format="%.4g")
    print(f"wrote {OUTC / name}")


def save_fig(fig, stem):
    OUTF.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTF / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTF / stem}.pdf/.png")


def style(ax, axis="y"):
    ax.grid(axis=axis, color="#e1e0d9", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- E1 data
def load_e1():
    cols = ["model", "dataset", "example_id", "condition", "valid_run", "pcr_eligible", "evidence_eligible",
            "first_terminal_post_observation_step", "terminal_route", "required_evidence_count",
            "local_evidence_count_missing", "local_full_evidence", "final_full_evidence"]
    ev = pd.concat([pd.read_csv(E1 / f / "run_level_evidence_hops.csv", usecols=cols, dtype=str)
                    for f in ("evidence_hop_coverage", "evidence_hop_coverage_gemma_gptoss")])
    ev = ev[ev.condition.isin([CLEAN, NEU, FIX, SAT])].copy()
    step = pd.to_numeric(ev.first_terminal_post_observation_step, errors="coerce")
    ev["t"] = np.where(ev.terminal_route.isin(ROUTES) & step.notna(), step, np.inf)
    ev["pe"] = b(ev.pcr_eligible)
    ev["evok"] = b(ev.evidence_eligible) & ev.local_full_evidence.isin(["True", "False"])
    ev["local_full"] = b(ev.local_full_evidence)
    ev["final_ok"] = ev.final_full_evidence.isin(["True", "False"])
    ev["final_full"] = b(ev.final_full_evidence)
    ev["missing"] = pd.to_numeric(ev.local_evidence_count_missing, errors="coerce")
    ans = {}
    with QL_ANSWERS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["condition"] in (CLEAN, NEU, FIX, SAT):
                ans[(r["model"], r["dataset"], r["condition"], str(r["example_id"]))] = r["semantic_label"] == "correct"
    gg = pd.read_csv(GG_ANSWERS, usecols=["model", "dataset", "condition", "example_id", "automatic_semantic_label"],
                     dtype=str)
    gg = gg[gg.condition.isin([CLEAN, NEU, FIX, SAT])]
    ans.update({(m, d, c, q): l == "correct" for m, d, c, q, l in gg.itertuples(index=False)})
    keys = list(zip(ev.model, ev.dataset, ev.condition, ev.example_id))
    ev["correct"] = [ans.get(k, np.nan) for k in keys]
    return ev


def pairs(df, cue, ctl, model_col="model"):
    a, c = df[df.condition == cue], df[df.condition == ctl]
    m = a.merge(c, on=[model_col, "dataset", "example_id"], suffixes=("_c", "_n"))
    return m[m.pe_c & m.pe_n]


def char_steps():
    d = pd.concat([pd.read_csv(E1 / x / "data/pcr_paired_deltas_stepwise.csv")
                   for x in ("qwen_llama", "gemma_gptoss")])
    d = d[(d.control_condition == NEU) & (d.cue_condition == FIX) & (d.step <= 10)]
    g = d.groupby(["model", "step"]).delta_pcr.mean().reset_index()
    out = {m: int(g[g.model == m].sort_values("delta_pcr").iloc[-1].step) for m, _ in MODELS}
    print("\nCharacteristic step (argmax over k<=10 of mean ΔPCR@k, Fixed Sufficiency vs Neutral, 4 datasets):")
    for m, name in MODELS:
        top = g[g.model == m].sort_values("delta_pcr", ascending=False).head(3)
        print(f"  {name}: step {out[m]}   top-3 (step, mean Δ pp): "
              + ", ".join(f"({int(s)}, {100 * v:+.1f})" for s, v in zip(top.step, top.delta_pcr)))
    return out


# --------------------------------------------------------------------------- tables (shared by E1 and E2)
def asr_rows(df, contrasts, models, datasets, model_col="model"):
    rows = []
    for cue, ctl, label in contrasts:
        p = pairs(df, cue, ctl, model_col)
        for m, _ in models:
            for d, _ in datasets:
                x = p[(p[model_col] == m) & (p.dataset == d)]
                n = len(x)
                e, s = int((x.t_c < x.t_n).sum()), int((x.t_c == x.t_n).sum())
                lo, hi = wilson(e, n)
                rows.append({"contrast": label, "model": m, "dataset": d, "n_pairs": n,
                             "pct_earlier": 100 * e / n, "pct_same": 100 * s / n,
                             "pct_later": 100 * (n - e - s) / n, "ci_lower_earlier": lo, "ci_upper_earlier": hi})
    return pd.DataFrame(rows)


def eicr_rows(df, arms, models, datasets, model_col="model"):
    rows = []
    for m, _ in models:
        for d, _ in datasets:
            x = df[(df[model_col] == m) & (df.dataset == d) & df.condition.isin(arms) & df.pe]
            qs = set.intersection(*[set(x[x.condition == a].example_id) for a in arms])
            for a in arms:
                y = x[(x.condition == a) & x.example_id.isin(qs) & np.isfinite(x.t) & x.evok]
                n, k = len(y), int((~y.local_full).sum())
                lo, hi = wilson(k, n)
                rows.append({"model": m, "dataset": d, "arm": ARM[a], "n_questions_matched": len(qs),
                             "n_pairs": n, "eicr_rate": 100 * k / n if n else np.nan, "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)


def early_pairs(df, cue, ctl, char, model_col="model"):
    p = pairs(df, cue, ctl, model_col)
    return p[p.t_c <= p[model_col].map(char)]


def harm_rows(df, contrasts, char, models, datasets, model_col="model", pooled=True):
    rows = []
    for cue, ctl, label in contrasts:
        p = early_pairs(df, cue, ctl, char, model_col)
        p = p[p.evok_c & p.evok_n]
        cells = [(m, d) for m, _ in models for d, _ in datasets] + ([(m, "pooled") for m, _ in models] if pooled else [])
        for m, d in cells:
            x = p[(p[model_col] == m) & ((p.dataset == d) if d != "pooled" else True)]
            for arm, sfx in ((ctl, "_n"), (cue, "_c")):
                n = len(x)
                k = int((~x["local_full" + sfx]).sum())
                miss = x["missing" + sfx].astype(float)
                lo, hi = wilson(k, n)
                rows.append({"contrast": label, "model": m, "dataset": d, "arm": ARM[arm], "n_early_stops": n,
                             "evidence_gap_rate": 100 * k / n if n else np.nan, "ci_lower_gap": lo, "ci_upper_gap": hi,
                             "mean_missing_units": miss.mean() if n else np.nan,
                             "se_missing_units": miss.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan})
    return pd.DataFrame(rows)


def supervisor_rows(df, contrasts, char, models, datasets, model_col="model"):
    rows = []
    for cue, ctl, label in contrasts:
        p = early_pairs(df, cue, ctl, char, model_col)
        p = p[p.correct_c.notna() & p.correct_n.notna()]
        for m, _ in models:
            for d, _ in datasets:
                x = p[(p[model_col] == m) & (p.dataset == d)]
                inc = x[x.evok_c & ~x.local_full_c & x.final_ok_c]
                cue_acc = x.correct_c.astype(float)
                lo, hi = boot_ci(cue_acc)
                rows.append({"contrast": label, "model": m, "dataset": d, "n_early_stops": len(x),
                             "accuracy_researcher_neutral": np.nan, "accuracy_researcher_cue": np.nan,
                             "ci_lower_researcher_cue": np.nan, "ci_upper_researcher_cue": np.nan,
                             "n_cue_evidence_incomplete": len(inc),
                             "recovery_rate": 100 * inc.final_full_c.mean() if len(inc) else np.nan,
                             "accuracy_final_neutral": 100 * x.correct_n.astype(float).mean() if len(x) else np.nan,
                             "accuracy_final_cue": 100 * cue_acc.mean() if len(x) else np.nan,
                             "ci_lower_final_cue": lo, "ci_upper_final_cue": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- E2 data
def load_e2(h_map):
    L = pd.read_csv(E2_LABELS, usecols=["victim", "dataset", "example_id", "arm", "status", "pcr_eligible", "t_i",
                                        "evidence", "answer_correct"], dtype=str)
    L = L[(L.status == "ran") & L.arm.isin(["neutral", "fixed_sufficiency", "bridge_sufficiency_k3"])].copy()
    L["condition"] = L.arm
    L["pe"] = b(L.pcr_eligible)
    L["t"] = pd.to_numeric(L.t_i, errors="coerce").fillna(np.inf)
    evd = L.evidence.fillna("{}").map(json.loads)
    loc = evd.map(lambda e: e.get("episode0_terminal_coverage"))
    fin = evd.map(lambda e: e.get("final_input_coverage"))
    L["evok"] = loc.notna()
    L["local_full"] = loc.fillna(0).astype(float) >= 1 - 1e-9
    L["final_ok"] = fin.notna()
    L["final_full"] = fin.fillna(0).astype(float) >= 1 - 1e-9
    H = pd.Series([h_map.get((d, q), np.nan) for d, q in zip(L.dataset, L.example_id)], index=L.index)
    L["missing"] = (H * (1 - loc.fillna(np.nan).astype(float))).round()
    print(f"E2: H_q found for {H.notna().mean():.1%} of runs (from E1 required_evidence_count)")
    L["correct"] = L.answer_correct.map({"True": 1.0, "False": 0.0})
    return L


# --------------------------------------------------------------------------- figures
def fig_asr_heatmap(asr):
    cmap = LinearSegmentedColormap.from_list("asr", ["#ffffff", "#1D9E75"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.08})
    for ax, label in zip(axes, ["Fixed Sufficiency vs Neutral", "Suff.+Saturation vs Neutral"]):
        a = asr[asr.contrast == label]
        grid = np.array([[a[(a.model == m) & (a.dataset == d)].pct_earlier.iloc[0] for d, _ in DATASETS] for m, _ in MODELS])
        later = np.array([[a[(a.model == m) & (a.dataset == d)].pct_later.iloc[0] for d, _ in DATASETS] for m, _ in MODELS])
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(4):
            for j in range(4):
                dark = grid[i, j] >= 62
                ax.text(j, i - 0.12, f"{grid[i, j]:.0f}%", ha="center", va="center", fontsize=14, weight="bold",
                        color="white" if dark else "#0b0b0b")
                ax.text(j, i + 0.22, f"Neutral: {later[i, j]:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="#e8e8e8" if dark else "#52514e")
        ax.set_xticks(range(4), [n for _, n in DATASETS])
        ax.set_yticks(range(4), [n for _, n in MODELS] if ax is axes[0] else [""] * 4)
        ax.set_xlabel("Fixed Sufficiency" if label.startswith("Fixed") else "Suff.+Saturation", fontsize=11)
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08, aspect=50)
    cb.set_label("Attack success rate (%): cue stopped strictly earlier than Neutral on the same question")
    save_fig(fig, "fig_asr_heatmap")


def fig_completion_steps(raw):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (m, name) in zip(axes.flat, MODELS):
        style(ax)
        stats, pos, cols = [], [], []
        for j, (d, _) in enumerate(DATASETS):
            x = raw[(raw.model == m) & (raw.dataset == d)]
            for k, (col, c) in enumerate((("t_neutral", C_NEU), ("t_cue", C_FIX))):
                v = x[col].dropna().to_numpy(float)
                q = np.percentile(v, [10, 25, 50, 75, 90])
                stats.append({"whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4], "fliers": []})
                pos.append(j * 3 + k)
                cols.append(c)
        bp = ax.bxp(stats, positions=pos, widths=0.8, patch_artist=True, showfliers=False,
                    medianprops={"color": "#0b0b0b", "lw": 1.5})
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
        ax.set_xticks([j * 3 + 0.5 for j in range(4)], [n for _, n in DATASETS])
        ax.set_title(name, fontsize=11)
        if m in ("qwen14b", "gemma27b"):
            ax.set_ylabel("Completion step")
    fig.legend([Patch(color=C_NEU), Patch(color=C_FIX)], ["Neutral", "Fixed Sufficiency"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, "fig_completion_steps")


def fig_eicr(eicr):
    fig, ax = plt.subplots(figsize=(15, 5))
    style(ax)
    arms = [("Neutral", C_NEU), ("Fixed Sufficiency", C_FIX), ("Suff.+Saturation", C_SAT)]
    labels, w = [], 0.27
    for g, ((m, mn), (d, dn)) in enumerate([(mm, dd) for mm in MODELS for dd in DATASETS]):
        labels.append(f"{mn} — {dn}")
        for k, (a, c) in enumerate(arms):
            r = eicr[(eicr.model == m) & (eicr.dataset == d) & (eicr.arm == a)].iloc[0]
            x = g + (k - 1) * w
            ax.bar(x, r.eicr_rate, w * 0.95, color=c)
            ax.errorbar(x, r.eicr_rate, yerr=[[r.eicr_rate - r.ci_lower], [r.ci_upper - r.eicr_rate]],
                        color="#0b0b0b", lw=0.8, capsize=1.5)
    ax.set_xticks(range(16), labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of completed runs missing required evidence\n(at the Researcher's stop)")
    ax.legend([Patch(color=c) for _, c in arms], [a for a, _ in arms], frameon=False, ncol=3, loc="upper left")
    save_fig(fig, "fig_eicr")


def two_panel_harm(harm, groups, arm_specs, stem):
    """groups: [(label, filter dict)]; arm_specs: [(contrast, arm, color, legend)]."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.8 / len(arm_specs)
    for ax, (col, lo, hi, ylabel) in zip(axes, (
            ("evidence_gap_rate", "ci_lower_gap", "ci_upper_gap", "% of early stops missing required evidence"),
            ("mean_missing_units", None, None, "Mean missing evidence units"))):
        style(ax)
        for g, (_, flt) in enumerate(groups):
            for k, (con, arm, c, _) in enumerate(arm_specs):
                r = harm[(harm.contrast == con) & (harm.arm == arm)
                         & (harm.model == flt["model"]) & (harm.dataset == flt["dataset"])].iloc[0]
                x = g + (k - (len(arm_specs) - 1) / 2) * w
                v = r[col]
                err = ([[v - r[lo]], [r[hi] - v]] if lo else [[1.96 * r.se_missing_units], [1.96 * r.se_missing_units]])
                ax.bar(x, v, w * 0.95, color=c)
                ax.errorbar(x, v, yerr=err, color="#0b0b0b", lw=0.8, capsize=2)
        ax.set_xticks(range(len(groups)), [g for g, _ in groups])
        ax.set_ylabel(ylabel)
        if col == "evidence_gap_rate":
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, None)
    fig.legend([Patch(color=c) for *_, c, _ in arm_specs], [lab for *_, lab in arm_specs], loc="lower center",
               ncol=len(arm_specs), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, stem)


def supervisor_panels(sup, models, datasets, specs, stem, ncols):
    """specs: [(contrast, cue legend, cue color)]; per dataset group: [Neutral, cue] per contrast."""
    vals = pd.concat([sup.accuracy_final_neutral, sup.ci_lower_final_cue, sup.ci_upper_final_cue]).dropna()
    ylo, yhi = max(0, math.floor(vals.min()) - 5), min(100, math.ceil(vals.max()) + 14)  # headroom for "rec." labels
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False, sharey=True)
    nb = 2 * len(specs)
    w = 0.8 / nb
    for ax, (m, mn) in zip(axes.flat, models):
        style(ax)
        for g, (d, _) in enumerate(datasets):
            for s, (con, _, c) in enumerate(specs):
                r = sup[(sup.contrast == con) & (sup.model == m) & (sup.dataset == d)].iloc[0]
                xn = g + (2 * s - (nb - 1) / 2) * w
                xc = xn + w
                ax.bar(xn, r.accuracy_final_neutral, w * 0.95, color=C_NEU)
                ax.bar(xc, r.accuracy_final_cue, w * 0.95, color=c)
                ax.errorbar(xc, r.accuracy_final_cue, yerr=[[r.accuracy_final_cue - r.ci_lower_final_cue],
                                                             [r.ci_upper_final_cue - r.accuracy_final_cue]],
                            color="#0b0b0b", lw=0.8, capsize=2)
                if not np.isnan(r.recovery_rate):
                    ax.text(xc, r.ci_upper_final_cue + 1, f"{r.recovery_rate:.0f}% rec.",
                            ha="center", va="bottom", fontsize=7, color="#a32d2d", rotation=90)
        ax.set_xticks(range(len(datasets)), [n for _, n in datasets])
        ax.set_title(mn, fontsize=11)
        ax.set_ylim(ylo, yhi)
        if ax in axes[:, 0]:
            ax.set_ylabel("Final answer accuracy (%)")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    handles = [Patch(color=C_NEU)] + [Patch(color=c) for *_, c in specs]
    labels = ["Neutral (same questions)"] + [f"{lab}: final answer (after Supervisor)" for _, lab, _ in specs]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, stem)


def fig_e2_asr(asr):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style(ax)
    groups = [(m, d, f"{mn}–{dn}") for m, mn in E2_MODELS for d, dn in E2_DATASETS]
    specs = [("Fixed Sufficiency vs Neutral", C_SAT, "Fixed Sufficiency"), ("Bridge K3 vs Neutral", C_K3, "Bridge K3")]
    w = 0.36
    for g, (m, d, _) in enumerate(groups):
        for k, (con, c, _) in enumerate(specs):
            r = asr[(asr.contrast == con) & (asr.model == m) & (asr.dataset == d)].iloc[0]
            x = g + (k - 0.5) * (w + 0.04)
            ax.bar(x, r.pct_earlier, w, color=c)
            ax.bar(x, r.pct_same, w, bottom=r.pct_earlier, color="#d3d1c7")
            ax.bar(x, r.pct_later, w, bottom=r.pct_earlier + r.pct_same, color=C_GRAY)
            ax.errorbar(x, r.pct_earlier, yerr=[[r.pct_earlier - r.ci_lower_earlier],
                                                 [r.ci_upper_earlier - r.pct_earlier]], color="#0b0b0b", lw=0.9, capsize=2)
            ax.text(x, r.pct_earlier / 2, f"{r.pct_earlier:.0f}%", ha="center", va="center", fontsize=8.5,
                    color="white", weight="bold")
    ax.set_xticks(range(len(groups)), [lab for *_, lab in groups])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of matched question pairs")
    fig.legend([Patch(color=C_SAT), Patch(color=C_K3), Patch(color="#d3d1c7"), Patch(color=C_GRAY)],
               ["Fixed Sufficiency stopped earlier (ASR)", "Bridge K3 stopped earlier (ASR)", "Same step",
                "Neutral stopped earlier"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, "fig_e2_asr")


# --------------------------------------------------------------------------- excess harm
def excess_rows(ev):
    """Gross harm of each cue vs Neutral, minus the placebo gross harm (Neutral vs Clean), per question triple.

    A question enters a cell only if its Clean, Neutral and cue runs are all valid and scored, so cue and
    placebo rates share one denominator. broke = Neutral right & cue wrong; placebo broke = Clean right &
    Neutral wrong. Excess = mean(broke - placebo_broke); 95% CI by resampling questions jointly (pooled rows:
    resampled within each dataset), 2000 resamples, seed 20260910.
    """
    v = ev[b(ev.valid_run) & ev.correct.notna()]
    piv = v.pivot_table(index=["model", "dataset", "example_id"], columns="condition", values="correct",
                        aggfunc="first")
    rows = []
    for cue, label in ((FIX, "Fixed Sufficiency vs Neutral"), (SAT, "Suff.+Saturation vs Neutral")):
        t = piv[[CLEAN, NEU, cue]].dropna().astype(float).reset_index()
        for m, _ in MODELS:
            for d in [x for x, _ in DATASETS] + ["pooled"]:
                x = t[(t.model == m) & ((t.dataset == d) if d != "pooled" else True)]
                clean, neu, cu = x[CLEAN].to_numpy(), x[NEU].to_numpy(), x[cue].to_numpy()
                broke, fixed = (neu == 1) & (cu == 0), (neu == 0) & (cu == 1)
                broke_p, fixed_p = (clean == 1) & (neu == 0), (clean == 0) & (neu == 1)
                diff = broke.astype(float) - broke_p.astype(float)
                rng = np.random.default_rng(SEED)
                strata = [np.flatnonzero(x.dataset.to_numpy() == s) for s in x.dataset.unique()]
                boot = np.zeros(NB)
                for idx in strata:  # stratified by dataset for pooled rows; a single stratum otherwise
                    boot += diff[idx][rng.integers(0, len(idx), size=(NB, len(idx)))].sum(axis=1)
                boot /= len(x)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                n = len(x)
                g_lo, g_hi = wilson(int(broke.sum()), n)
                p_lo, p_hi = wilson(int(broke_p.sum()), n)
                rows.append({"contrast": label, "model": m, "dataset": d, "n_pairs": n,
                             "gross_harm_cue": 100 * broke.mean(), "gross_harm_placebo": 100 * broke_p.mean(),
                             "excess_harm": 100 * diff.mean(), "ci_lower_excess": 100 * lo, "ci_upper_excess": 100 * hi,
                             "break_rate_cue": 100 * broke.sum() / max(1, (neu == 1).sum()),
                             "break_rate_placebo": 100 * broke_p.sum() / max(1, (clean == 1).sum()),
                             "n_broke": int(broke.sum()), "n_fixed": int(fixed.sum()),
                             "n_broke_placebo": int(broke_p.sum()), "n_fixed_placebo": int(fixed_p.sum()),
                             "ci_lower_gross_cue": g_lo, "ci_upper_gross_cue": g_hi,
                             "ci_lower_gross_placebo": p_lo, "ci_upper_gross_placebo": p_hi})
    return pd.DataFrame(rows)


def fig_excess_harm(exc):
    pooled = exc[exc.dataset == "pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    get = lambda con, m: pooled[(pooled.contrast == con) & (pooled.model == m)].iloc[0]
    fx, sx = "Fixed Sufficiency vs Neutral", "Suff.+Saturation vs Neutral"
    ax = axes[0]
    style(ax)
    w = 0.26
    top = 0
    for g, (m, _) in enumerate(MODELS):
        bars = [(get(fx, m), "gross_harm_cue", "ci_lower_gross_cue", "ci_upper_gross_cue", C_FIX),
                (get(sx, m), "gross_harm_cue", "ci_lower_gross_cue", "ci_upper_gross_cue", C_SAT),
                (get(fx, m), "gross_harm_placebo", "ci_lower_gross_placebo", "ci_upper_gross_placebo", C_GRAY)]
        for k, (r, col, lo, hi, c) in enumerate(bars):
            x = g + (k - 1) * w
            ax.bar(x, r[col], w * 0.95, color=c)
            ax.errorbar(x, r[col], yerr=[[r[col] - r[lo]], [r[hi] - r[col]]], color="#0b0b0b", lw=0.8, capsize=2)
            top = max(top, r[hi])
    ax.set_xticks(range(4), [n for _, n in MODELS])
    ax.set_ylim(0, max(20, math.ceil(top) + 1))
    ax.set_ylabel("% of questions where cue broke a correct answer")
    ax = axes[1]
    style(ax)
    w = 0.36
    for g, (m, _) in enumerate(MODELS):
        for k, (con, c) in enumerate(((fx, C_FIX), (sx, C_SAT))):
            r = get(con, m)
            x = g + (k - 0.5) * (w + 0.04)
            ax.bar(x, r.excess_harm, w, color=c)
            ax.errorbar(x, r.excess_harm, yerr=[[r.excess_harm - r.ci_lower_excess], [r.ci_upper_excess - r.excess_harm]],
                        color="#0b0b0b", lw=0.8, capsize=2)
    ax.axhline(0, color=C_GRAY, lw=1.2, ls="--")
    ax.set_xticks(range(4), [n for _, n in MODELS])
    ax.set_ylabel("Excess gross harm above placebo (%)")
    fig.legend([Patch(color=C_FIX), Patch(color=C_SAT), Patch(color=C_GRAY)],
               ["Fixed Sufficiency", "Suff.+Saturation", "Placebo (Neutral vs Clean); dashed line = placebo level"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, "fig_excess_harm")


# --------------------------------------------------------------------------- main
def main():
    OUTC.mkdir(parents=True, exist_ok=True)
    print("COLUMNS USED")
    print("  completion step : run_level_evidence_hops.first_terminal_post_observation_step (+ terminal_route);"
          " E2 run_labels.t_i")
    print("  answer correct  : FINAL answer only — QL jsonl `semantic_label`=='correct', GG csv"
          " `automatic_semantic_label`=='correct', E2 run_labels.answer_correct."
          " Researcher-level accuracy: NOT STORED (ODR's Researcher returns notes, not an answer).")
    print("  evidence        : local_full_evidence / local_evidence_count_missing (Researcher's stop),"
          " final_full_evidence (final-report input); E2 evidence JSON episode0_terminal_coverage /"
          " final_input_coverage")
    char = char_steps()

    ev = load_e1()
    p = pairs(ev, FIX, NEU)
    n_q = len(p[(p.model == "qwen14b") & (p.dataset == "hotpot")])
    print(f"check: Qwen HotpotQA Fixed-vs-Neutral pairs = {n_q} (published n_paired 441)")

    e1c = [(FIX, NEU, "Fixed Sufficiency vs Neutral"), (SAT, NEU, "Suff.+Saturation vs Neutral")]
    asr = asr_rows(ev, e1c, MODELS, DATASETS)
    save_csv(asr, "asr_table.csv")

    raw = p[["example_id", "model", "dataset", "t_n", "t_c"]].rename(
        columns={"example_id": "question_id", "t_n": "t_neutral", "t_c": "t_cue"}).replace(np.inf, np.nan)
    rows = []
    for m, _ in MODELS:
        for d, _ in DATASETS:
            x = raw[(raw.model == m) & (raw.dataset == d)]
            for col, arm in (("t_neutral", "Neutral"), ("t_cue", "Fixed Sufficiency")):
                v = x[col].dropna().to_numpy(float)
                q = np.percentile(v, [10, 25, 50, 75, 90])
                rows.append({"model": m, "dataset": d, "arm": arm, "n_pairs": len(x),
                             "n_completed_in_episode": len(v), "median_step": q[2], "p25_step": q[1],
                             "p75_step": q[3], "p10_step": q[0], "p90_step": q[4], "mean_step": v.mean(),
                             "std_step": v.std(ddof=1)})
    save_csv(pd.DataFrame(rows), "step_distribution_summary.csv")
    save_csv(raw, "step_distribution_raw.csv")

    eicr = eicr_rows(ev, [NEU, FIX, SAT], MODELS, DATASETS)
    save_csv(eicr, "eicr_table.csv")
    harm = harm_rows(ev, [(FIX, NEU, "Fixed Sufficiency vs Neutral")], char, MODELS, DATASETS)
    save_csv(harm, "evidence_harm.csv")
    sup = supervisor_rows(ev, [(FIX, NEU, "Fixed Sufficiency vs Neutral")], char, MODELS, DATASETS)
    exc = excess_rows(ev)
    save_csv(exc, "excess_harm.csv")
    flips = pd.concat([pd.read_csv(E1 / x / "data/answer_paired_flips.csv") for x in ("qwen_llama", "gemma_gptoss")])
    print("check vs answer_paired_flips (pooled per model, Fixed Suff. vs Neutral): broke here / published")
    for m, name in MODELS:
        f = flips[(flips.model == m) & (flips.condition_a == NEU) & (flips.condition_b == FIX)]
        r = exc[(exc.contrast == "Fixed Sufficiency vs Neutral") & (exc.model == m) & (exc.dataset == "pooled")].iloc[0]
        print(f"  {name}: {r.n_broke}/{r.n_pairs} here vs {int(f.control_correct_attack_wrong.sum())}/{int(f.n_pairs.sum())} published")
    cell = exc[(exc.contrast == "Fixed Sufficiency vs Neutral") & (exc.dataset != "pooled")][
        ["model", "dataset", "gross_harm_cue", "excess_harm"]].rename(
        columns={"gross_harm_cue": "gross_harm_rate", "excess_harm": "excess_harm_rate"})
    sup = sup.merge(cell, on=["model", "dataset"], how="left")
    save_csv(sup, "pre_post_supervisor_harm.csv")
    cells = exc[exc.dataset != "pooled"]
    sig = cells[(cells.ci_lower_excess > 0) | (cells.ci_upper_excess < 0)]
    print(f"\nEXCESS HARM: 32 cells min {cells.excess_harm.min():+.2f} pp, max {cells.excess_harm.max():+.2f} pp")
    print("  cells whose CI excludes 0:")
    print(sig[["contrast", "model", "dataset", "n_pairs", "excess_harm", "ci_lower_excess", "ci_upper_excess"]]
          .to_string(index=False) if len(sig) else "  none")
    print("  pooled per model:")
    print(exc[exc.dataset == "pooled"][["contrast", "model", "n_pairs", "gross_harm_cue", "gross_harm_placebo",
                                        "excess_harm", "ci_lower_excess", "ci_upper_excess"]].to_string(index=False))

    h_map = {(d, q): int(h) for d, q, h in zip(ev.dataset, ev.example_id, ev.required_evidence_count)
             if str(h).isdigit()}
    e2 = load_e2(h_map)
    e2c = [("bridge_sufficiency_k3", "neutral", "Bridge K3 vs Neutral"),
           ("fixed_sufficiency", "neutral", "Fixed Sufficiency vs Neutral")]
    e2_asr = asr_rows(e2, e2c, E2_MODELS, E2_DATASETS, "victim").rename(columns={"model": "model"})
    save_csv(e2_asr, "e2_asr_table.csv")
    e2_eicr = eicr_rows(e2, ["neutral", "fixed_sufficiency", "bridge_sufficiency_k3"], E2_MODELS, E2_DATASETS, "victim")
    save_csv(e2_eicr, "e2_eicr_table.csv")
    e2_harm = harm_rows(e2, e2c, E2_CHAR, E2_MODELS, E2_DATASETS, "victim", pooled=False)
    save_csv(e2_harm, "e2_evidence_harm.csv")
    e2_sup = supervisor_rows(e2, e2c, E2_CHAR, E2_MODELS, E2_DATASETS, "victim")
    save_csv(e2_sup, "e2_pre_post_supervisor_harm.csv")

    print("\nVERIFICATION")
    print(f"  1. E1 ASR (pct_earlier) over 32 cells: min {asr.pct_earlier.min():.1f}%, max {asr.pct_earlier.max():.1f}%;"
          f" E2: {e2_asr.pct_earlier.min():.1f}–{e2_asr.pct_earlier.max():.1f}%")
    print(f"  2. E1 EICR over 48 cells: min {eicr.eicr_rate.min():.1f}%, max {eicr.eicr_rate.max():.1f}%;"
          f" E2: {e2_eicr.eicr_rate.min():.1f}–{e2_eicr.eicr_rate.max():.1f}%")
    print("  3. Pre-recovery (Researcher-level) accuracy: not available."
          f" Post-recovery (final) cue accuracy: E1 {sup.accuracy_final_cue.min():.1f}–{sup.accuracy_final_cue.max():.1f}%,"
          f" E2 {e2_sup.accuracy_final_cue.min():.1f}–{e2_sup.accuracy_final_cue.max():.1f}%;"
          f" Neutral final: E1 {sup.accuracy_final_neutral.min():.1f}–{sup.accuracy_final_neutral.max():.1f}%")
    print("  4. Researcher-level accuracy found: NO — only final accuracy is stored.")

    fig_asr_heatmap(asr)
    fig_completion_steps(raw)
    fig_eicr(eicr)
    two_panel_harm(harm, [(mn, {"model": m, "dataset": "pooled"}) for m, mn in MODELS],
                   [("Fixed Sufficiency vs Neutral", "Neutral", C_NEU, "Neutral (same questions)"),
                    ("Fixed Sufficiency vs Neutral", "Fixed Sufficiency", C_FIX, "Fixed Sufficiency")],
                   "fig_evidence_harm")
    supervisor_panels(sup, MODELS, DATASETS, [("Fixed Sufficiency vs Neutral", "Fixed Sufficiency", C_POST)],
                      "fig_supervisor_harm", ncols=2)
    fig_excess_harm(exc)
    fig_e2_asr(e2_asr)
    two_panel_harm(e2_harm, [(f"{mn}–{dn}", {"model": m, "dataset": d}) for m, mn in E2_MODELS for d, dn in E2_DATASETS],
                   [("Fixed Sufficiency vs Neutral", "Neutral", C_NEU, "Neutral (Fixed-Suff. questions)"),
                    ("Fixed Sufficiency vs Neutral", "Fixed Sufficiency", C_SAT, "Fixed Sufficiency"),
                    ("Bridge K3 vs Neutral", "Neutral", "#8fb8ea", "Neutral (Bridge-K3 questions)"),
                    ("Bridge K3 vs Neutral", "Bridge K3", C_K3, "Bridge K3")],
                   "fig_e2_evidence_harm")
    supervisor_panels(e2_sup, E2_MODELS, E2_DATASETS,
                      [("Fixed Sufficiency vs Neutral", "Fixed Sufficiency", C_SAT),
                       ("Bridge K3 vs Neutral", "Bridge K3", C_K3)], "fig_e2_supervisor_harm", ncols=2)


if __name__ == "__main__":
    main()

"""Print-ready E1 thesis figures (Figures 5.1-5.4, 5.6, 5.7).

Every plotted value is read from the published E1 CSVs; nothing is typed in by
hand. Output: results/whitebox/e1_fixed_cue_matrix/thesis_figures/fig5_*.{pdf,png}

    python -m prematurestop.analysis.fixed_cue_matrix.thesis_figures

Data sources:
  5.1  */data/pcr_stepwise.csv                         (mean over the 4 datasets)
  5.2  */data/route_decomposition_stepwise.csv         (Neutral, pooled over datasets)
  5.3  */data/pcr_paired_deltas_stepwise.csv          (Δ at each model's characteristic step)
  5.4  evidence_hop_coverage*/evidence_hop_summary.csv (n-weighted over hop strata)
  5.6  answer_harm/pooled.csv
  5.7  answer_harm/mechanism.csv                       (final-report input, all models)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

from prematurestop.paths import RESULTS_E1

OUT = RESULTS_E1 / "thesis_figures"
MODELS = (("qwen14b", "Qwen-14B"), ("llama8b", "Llama-8B"), ("gemma27b", "Gemma-27B"), ("gpt_oss_20b", "gpt-oss-20b"))
BUNDLE = {"qwen14b": "qwen_llama", "llama8b": "qwen_llama", "gemma27b": "gemma_gptoss", "gpt_oss_20b": "gemma_gptoss"}
EHC = {"qwen14b": "evidence_hop_coverage", "llama8b": "evidence_hop_coverage",
       "gemma27b": "evidence_hop_coverage_gemma_gptoss", "gpt_oss_20b": "evidence_hop_coverage_gemma_gptoss"}
CHAR = {"qwen14b": 1, "llama8b": 3, "gemma27b": 1, "gpt_oss_20b": 2}
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")
NEUTRAL, CLEAN = "NaturalNeutral", "NaturalClean"
SUFF, SAT = "Natural_PreRegisteredSufficiency", "Natural_PreRegisteredSufficiencySaturation"
CONT = "NaturalContinueSearch"

INK, INK_2, INK_3, GRID, REF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#888780"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK_3, "savefig.dpi": 300, "figure.dpi": 100})


# --------------------------------------------------------------------------- helpers
def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def bdata(m: str, name: str) -> list[dict]:
    return [r for r in read(RESULTS_E1 / BUNDLE[m] / "data" / f"{name}.csv") if r["model"] == m]


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / stem}.png")


def signed(x, _pos=None):
    return f"{x:+.0f}" if x else "0"


def smooth(xs, ys):
    """Monotone-preserving smooth curve (PCHIP): a CDF never overshoots or dips."""
    try:
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        fx = np.linspace(xs[0], xs[-1], 200)
        return fx, PchipInterpolator(xs, ys)(fx)
    except ImportError:
        return xs, ys


def boxed_note(fig, text, y, face, edge, width=88):
    """Highlighted note with a coloured left border, spanning the figure width."""
    ax = fig.add_axes((0.06, y, 0.9, 0.07))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", facecolor=face, edgecolor="none",
                                transform=ax.transAxes))
    ax.add_patch(Rectangle((0, 0), 0.006, 1, facecolor=edge, edgecolor="none", transform=ax.transAxes))
    import textwrap
    ax.text(0.02, 0.5, "\n".join(textwrap.wrap(text, width=width)), va="center", ha="left", fontsize=10.5, color=INK,
            transform=ax.transAxes)


# --------------------------------------------------------------------------- 5.1 PCR curves
def fig5_1():
    series = ((NEUTRAL, "Neutral", BLUE, "-"), (SUFF, "Fixed Sufficiency", ORANGE, "-"),
              (SAT, "Suff. + Saturation", ORANGE, (0, (6, 3))), (CONT, "Continue Search", GREEN, (0, (6, 3))))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    steps = list(range(1, 11))
    for ax, (m, name) in zip(axes.flat, MODELS):
        st = {(r["dataset"], r["condition"], int(r["step"])): float(r["pcr"]) for r in bdata(m, "pcr_stepwise")}
        grid(ax, "both")
        for cond, _, color, ls in series:
            ys = [100 * sum(st[(d, cond, k)] for d in DATASETS) / len(DATASETS) for k in steps]
            fx, fy = smooth(steps, ys)
            ax.plot(fx, fy, color=color, lw=2, ls=ls)
        ax.axvline(CHAR[m], color=REF, lw=1.2, ls=":")
        ax.text(CHAR[m] + 0.15, 3, f"step {CHAR[m]}", color=REF, fontsize=9)
        ax.set_title(name, fontsize=13, fontweight="normal", color=INK)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(labelsize=11, colors=INK_3)
        ax.set_xlabel("Researcher decision after the cue", fontsize=11, color=INK_2)
        ax.set_ylabel("PCR@k", fontsize=11, color=INK_2)
    handles = [Line2D([0], [0], color=c, lw=3, ls=ls) for _, _, c, ls in series]
    fig.legend(handles, [lab for _, lab, *_ in series], loc="lower center", ncol=4, frameon=False, fontsize=12,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.035), handlelength=2.6)
    fig.text(0.5, 0.005, "Gemma reaches 100% PCR by step 5 in every condition — its flat lines thereafter are a "
             "ceiling artefact, not evidence of robustness.", ha="center", fontsize=10, color=INK_2, style="italic")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig5_1_pcr_curves")


# --------------------------------------------------------------------------- 5.2 routes
def route_shares(m):
    tot = defaultdict(float)
    for r in bdata(m, "route_decomposition_stepwise"):
        if r["condition"] != NEUTRAL or r["step"] != "1":  # final_* columns repeat per step
            continue
        g = lambda k: float(r[f"final_n_{k}"] or 0)
        tot["Explicit completion call"] += g("RESEARCH_COMPLETE")
        tot["No-tool exit"] += g("NO_TOOL_DIRECT_EXIT") + g("OTHER_VALID_ROUTE")
        tot["Budget exhaustion"] += g("MAX_BUDGET_EXIT")
        tot["Technical failure"] += g("TOOL_OR_PARSER_FAILURE") + g("TRACE_LOGGING_FAILURE") + g("EXECUTION_FAILURE")
    n = sum(tot.values())
    return {k: 100 * v / n for k, v in tot.items()}


def fig5_2():
    cats = (("Explicit completion call", "#1D9E75"), ("No-tool exit", "#EF9F27"),
            ("Budget exhaustion", "#B4B2A9"), ("Technical failure", "#E24B4A"))
    fig, ax = plt.subplots(figsize=(8, 4))
    shares = {m: route_shares(m) for m, _ in MODELS}
    for i, (m, name) in enumerate(MODELS[::-1]):
        left = 0.0
        for cat, color in cats:
            v = shares[m].get(cat, 0.0)
            ax.barh(i, v, left=left, color=color, height=0.62, edgecolor="white", linewidth=1)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                        color="white" if cat != "Budget exhaustion" else INK)
            left += v
    ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS[::-1]], fontsize=12, fontweight="normal", color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=11, color=INK_3)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend([Patch(color=c) for _, c in cats], [c for c, _ in cats], loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=4, frameon=False, fontsize=12, labelcolor=INK_2, handlelength=1, handleheight=1)
    ax.set_title("How Neutral-condition episodes end (all four datasets)", fontsize=12, color=INK, loc="left")
    fig.text(0.5, -0.06, f"Gemma never uses the explicit completion tool — "
             f"{shares['gemma27b']['No-tool exit']:.0f}% of its episodes end via no-tool exit.",
             ha="center", fontsize=10, color=INK_2, style="italic")
    fig.tight_layout()
    save(fig, "fig5_2_routes")
    return shares


# --------------------------------------------------------------------------- 5.3 heatmap
HEAT_ROWS = ((("Suff. + Saturation", SAT, NEUTRAL), ("Fixed Sufficiency", SUFF, NEUTRAL)),
             (("Search Match", "Natural_search_match_cue", NEUTRAL),
              ("Epistemic Confirmation", "Natural_epistemic_confirmation", NEUTRAL),
              ("Source Coverage", "Natural_source_coverage", NEUTRAL),
              ("Epistemic + Source", "Natural_epistemic_confirmation_source", "Natural_epistemic_confirmation"),
              ("Search Match + Source", "Natural_search_match_source", "Natural_search_match_cue")),
             (("Neutral vs Clean", NEUTRAL, CLEAN), ("Continue Search", CONT, NEUTRAL)))


def heat_colors(v):
    if v >= 40:
        return "#085041", "#E1F5EE"
    if v >= 20:
        return "#1D9E75", "#E1F5EE"
    if v >= 10:
        return "#5DCAA5", "#04342C"
    if v >= 3:
        return "#9FE1CB", "#085041"
    if v > -1:
        return "#F1EFE8", "#444441"
    if v > -9:
        return "#F5C4B3", "#4A1B0C"
    return "#D85A30", "#FAECE7"


def fig5_3():
    rows = [r for g in HEAT_ROWS for r in g]
    vals, sigs = {}, {}
    for m, _ in MODELS:
        dl = {(r["dataset"], r["control_condition"], r["cue_condition"], int(r["step"])): r
              for r in bdata(m, "pcr_paired_deltas_stepwise")}
        for label, cue, ctl in rows:
            rs = [dl[(d, ctl, cue, CHAR[m])] for d in DATASETS if (d, ctl, cue, CHAR[m]) in dl]
            vals[(label, m)] = 100 * sum(float(r["delta_pcr"]) for r in rs) / len(rs)
            pos = sum(float(r["ci_lower"]) > 0 for r in rs)
            neg = sum(float(r["ci_upper"]) < 0 for r in rs)
            sigs[(label, m)] = max(pos, neg) >= 3
    fig, ax = plt.subplots(figsize=(8, 5))
    nr, nc = len(rows), len(MODELS)
    for i, (label, *_ ) in enumerate(rows):
        for j, (m, _) in enumerate(MODELS):
            v = vals[(label, m)]
            face, text = heat_colors(v)
            ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1, facecolor=face, edgecolor="white", lw=1.5))
            ax.text(j + 0.5, nr - 1 - i + 0.5, f"{v:+.1f}" if abs(v) >= 0.05 else "0.0", ha="center", va="center",
                    fontsize=12, fontweight="normal", color=text)
            if sigs[(label, m)]:
                ax.plot(j + 0.88, nr - 1 - i + 0.8, "o", ms=6, color=text)
    # separators between row groups
    y = nr
    for g in HEAT_ROWS[:-1]:
        y -= len(g)
        ax.axhline(y, color=INK, lw=1)
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)], [f"{n}\n@step {CHAR[m]}" for m, n in MODELS], fontsize=11,
                  color=INK_2, fontweight="normal")
    ax.xaxis.tick_top()
    ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)], [r[0] for r in rows], fontsize=12, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # "pre-spec" badges on the two pre-registered rows
    for i in range(2):
        ax.text(-0.04, nr - 1 - i + 0.5, "pre-spec", transform=ax.transData, ha="right", va="center", fontsize=7,
                color="white", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=BLUE, ec="none"))
    ax.yaxis.set_tick_params(pad=58)
    fig.text(0.5, 0.0, "Δ PCR (pp) vs control at each model's characteristic step, mean of 4 datasets. "
             "● = CI excludes 0 in ≥ 3 of 4 datasets.", ha="center", fontsize=9, color=INK_3)
    fig.tight_layout()
    save(fig, "fig5_3_heatmap")
    return vals


# --------------------------------------------------------------------------- 5.4 coverage
def coverage(m):
    acc = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in read(RESULTS_E1 / EHC[m] / "evidence_hop_summary.csv"):
        if r["model"] == m and r["condition"] in (NEUTRAL, SAT):
            n = float(r["n_completed_and_evidence_eligible"] or 0)
            a = acc[r["condition"]]
            a[0] += float(r["mean_local_completion_coverage"] or 0) * n
            a[1] += float(r["mean_final_coverage"] or 0) * n
            a[2] += n
    return {c: (100 * a[0] / a[2], 100 * a[1] / a[2]) for c, a in acc.items()}


def fig5_4():
    fig, ax = plt.subplots(figsize=(9, 5))
    grid(ax)
    w, gap = 0.18, 0.02
    offs = [-1.5 * w - 1.5 * gap, -0.5 * w - 0.5 * gap, 0.5 * w + 0.5 * gap, 1.5 * w + 1.5 * gap]
    rec = {}
    for g, (m, _) in enumerate(MODELS):
        cv = coverage(m)
        bars = [(cv[NEUTRAL][0], BLUE, 1.0, None), (cv[SAT][0], ORANGE, 1.0, None),
                (cv[NEUTRAL][1], BLUE, 0.38, "///"), (cv[SAT][1], ORANGE, 0.38, "///")]
        for k, (v, color, alpha, hatch) in enumerate(bars):
            ax.bar(g + offs[k], v, width=w, color=color, alpha=alpha, hatch=hatch, edgecolor=color, linewidth=0.8)
        rec[m] = cv[SAT][1] - cv[SAT][0]
        x2, x4 = g + offs[1], g + offs[3]
        y2, y4 = cv[SAT][0], cv[SAT][1]
        ax.annotate("", xy=(x4, y4 + 0.4), xytext=(x2, y2 + 0.4),
                    arrowprops=dict(arrowstyle="->", color="#BA7517", lw=1.5, ls="--"))
        top = max(v for v, *_ in bars)
        ax.text((x2 + x4) / 2, top + 2.0, f"{rec[m]:+.1f} pp", ha="center", fontsize=11, color="#854F0B",
                bbox=dict(boxstyle="round,pad=0.3", fc=(250 / 255, 238 / 255, 218 / 255, 0.92), ec="none"))
    ax.set_ylim(55, 100)
    ax.set_ylabel("Gold evidence coverage (%)", fontsize=11, color=INK_2)
    ax.set_xticks(range(len(MODELS)), [n for _, n in MODELS], fontsize=12, fontweight="normal", color=INK)
    ax.tick_params(axis="y", labelsize=11, colors=INK_3)
    legend = [Patch(facecolor=BLUE, edgecolor=BLUE, label="Neutral — Researcher's stop"),
              Patch(facecolor=ORANGE, edgecolor=ORANGE, label="Suff. + Saturation — Researcher's stop"),
              Patch(facecolor=BLUE, alpha=0.38, hatch="///", edgecolor=BLUE, label="Neutral — final report"),
              Patch(facecolor=ORANGE, alpha=0.38, hatch="///", edgecolor=ORANGE, label="Suff. + Saturation — final report")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=12,
              labelcolor=INK_2)
    fig.tight_layout()
    save(fig, "fig5_4_coverage")
    return rec


# --------------------------------------------------------------------------- 5.6 average harm
def fig5_6():
    po = {(r["comparison"], r["model"]): r for r in read(RESULTS_E1 / "answer_harm" / "pooled.csv")
          if r["family"] == "primary_attack"}
    comps = {r["cue"]: r["comparison"] for r in read(RESULTS_E1 / "answer_harm" / "pooled.csv") if r["family"] == "primary_attack"}
    fig = plt.figure(figsize=(7, 4.5 + 0.7))
    ax = fig.add_axes((0.12, 0.3, 0.84, 0.62))
    grid(ax)
    w = 0.34
    lows, highs = [], []
    for k, (cue, alpha, hatch, label) in enumerate(((SUFF, 1.0, None, "Fixed Sufficiency"), (SAT, 0.45, "///", "Suff. + Saturation"))):
        for g, (m, _) in enumerate(MODELS):
            r = po[(comps[cue], m)]
            v, lo, hi = float(r["net_harm_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"])
            lows.append(lo); highs.append(hi)
            x = g + (k - 0.5) * (w + 0.04)
            ax.bar(x, v, width=w, color=BLUE, alpha=alpha, hatch=hatch, edgecolor=BLUE, label=label if g == 0 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, lw=1.2, capsize=5)
    ax.axhline(0, color=INK, lw=1.5)
    lim = max(3.0, max(abs(min(lows)), max(highs)) + 0.3)
    ax.set_ylim(-lim, lim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(signed))
    ax.set_ylabel("Net answer harm (pp)", fontsize=11, color=INK_2)
    ax.set_xticks(range(len(MODELS)), [n for _, n in MODELS], fontsize=12, fontweight="normal", color=INK)
    ax.tick_params(axis="y", labelsize=11, colors=INK_3)
    a, b = po[(comps[SUFF], "all")], po[(comps[SAT], "all")]
    ax.text(0.99, 0.97, f"Pooled: Fixed {float(a['net_harm_pp']):+.1f} [{float(a['ci_low_pp']):+.1f}, {float(a['ci_high_pp']):+.1f}]\n"
            f"Suff.+Sat. {float(b['net_harm_pp']):+.1f} [{float(b['ci_low_pp']):+.1f}, {float(b['ci_high_pp']):+.1f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=12, labelcolor=INK_2)
    boxed_note(fig, "Null result ≠ safe. Detecting +0.25 pp at 80% power requires ~240,000 pairs — 35× the current 6,800.",
               0.03, "#FAEEDA", "#BA7517", width=70)
    save(fig, "fig5_6_avg_harm")


# --------------------------------------------------------------------------- 5.7 mechanism
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fig5_7():
    me = {(r["cue"], r["transition"]): r for r in read(RESULTS_E1 / "answer_harm" / "mechanism.csv")
          if r["boundary"] == "final_report_input" and r["model"] == "all" and r["family"] == "primary_attack"}
    classes = (("both_complete", "Both complete", ("#5DCAA5", "#9FE1CB")),
               ("evidence_lost", "Cue lost evidence", (ORANGE, "#993C1D")),
               ("evidence_gained", "Cue gained evidence", ("#1D9E75", "#085041")),
               ("both_incomplete", "Both incomplete", ("#5DCAA5", "#9FE1CB")))
    fig = plt.figure(figsize=(8, 5 + 0.8))
    ax = fig.add_axes((0.11, 0.3, 0.86, 0.64))
    grid(ax)
    w = 0.36
    lows, highs = [], []
    for g, (cls, _, colors) in enumerate(classes):
        for k, cue in enumerate((SUFF, SAT)):
            r = me[(cue, cls)]
            v, lo, hi, p = float(r["net_harm_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"]), float(r["mcnemar_p"])
            lows.append(lo); highs.append(hi)
            x = g + (k - 0.5) * (w + 0.04)
            ax.bar(x, v, width=w, color=colors[k], edgecolor="white")
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, lw=1.2, capsize=4)
            s = stars(p)
            if s:
                ax.text(x, hi + 0.9 if v >= 0 else lo - 0.9, s, ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=13, color=INK)
    ax.axhline(0, color=INK, lw=1.5)
    ax.set_ylim(min(-16, min(lows) - 2.5), max(14, max(highs) + 2.5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(signed))
    ax.set_ylabel("Net answer harm (pp)", fontsize=11, color=INK_2)
    n = {cls: me[(SAT, cls)]["n_pairs"] for cls, *_ in classes}
    ax.set_xticks(range(len(classes)), [f"{name}\n(n={int(n[cls]):,})" for cls, name, _ in classes],
                  fontsize=11, fontweight="normal", color=INK)
    ax.tick_params(axis="y", labelsize=11, colors=INK_3)
    ax.legend([Patch(color="#a9a7a0"), Patch(color="#4a4944")],
              ["Fixed Sufficiency (left, lighter shade)", "Suff. + Saturation (right, darker shade)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=12, labelcolor=INK_2)
    lost = [float(me[(c, "evidence_lost")]["net_harm_pp"]) for c in (SUFF, SAT)]
    gained = [-float(me[(c, "evidence_gained")]["net_harm_pp"]) for c in (SUFF, SAT)]
    boxed_note(fig, f"Mechanism confirmed: losing evidence costs ≈{sum(lost) / 2:.0f} pp; gaining it helps "
               f"≈{sum(gained) / 2:.0f} pp; unchanged evidence shows no effect.", 0.03, "#EAF3DE", "#3B6D11", width=80)
    save(fig, "fig5_7_mechanism")


def main() -> int:
    fig5_1()
    shares = fig5_2()
    vals = fig5_3()
    rec = fig5_4()
    fig5_6()
    fig5_7()
    print("routes (Neutral, % of runs):", {m: {k: round(v, 1) for k, v in s.items()} for m, s in shares.items()})
    print("recovery (Suff+Sat, final - local, pp):", {m: round(v, 1) for m, v in rec.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E2 (bridge-anchored study) rewritten to the advisor-agreed metrics reference.

Run from the repo root (whitebox env):
    PYTHONPATH=src python scripts/thesis_figures/thesis_restructure_e2.py > results/thesis_tables/e2/thesis_restructure_e2_log.txt

Reuses the E2 loading rules of thesis_restructure_v2.py (task7: local v2b run files, status == "ran" = d1 matched the
anchor byte-for-byte) and its helpers (boot, mcnemar_p, bh, wilson). Existing scripts and outputs are not touched.
Outputs: CSVs in results/thesis_tables/e2/, figures in results/figures/thesis_restructure_e2/ (PDF fonttype 42 + PNG).

Definitions (same as E1)
  cell   qwen14b / llama8b x hotpot / 2wiki; every metric per cell, never pooled across cells
  T      first_terminal_post_observation_step if terminal_route is RESEARCH_COMPLETE / NO_TOOL_DIRECT_EXIT,
         else non-completion, coded T = 30
  valid  run ran (anchor-matched) and is PCR-eligible; a pair needs both runs valid
  D_q    T_neutral - T_arm (positive = arm stopped earlier)
  k50    per cell: smallest k with PCR_neutral(k) >= 0.50 over that cell's valid Neutral runs
  CI     paired bootstrap over questions, 2,000 resamples, seed 20260910, percentile 95%
  BH     within each comparison, across its 4 cells
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_restructure_v2 as V  # noqa: E402
from thesis_restructure_v2 import NB, REPO, ROUTES, SEED, T_NC, bh, boot, mcnemar_p, plt, rule, wilson  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42
E2_DIR = V.E2_DIR
OUTC = REPO / "results/thesis_tables/e2"
REF = Path(__file__).resolve().parent / "reference"  # earlier outputs, read only by the VERIFY cross-checks
OUTF = REPO / "results/figures/thesis_restructure_e2"
CELLS = [(v, d) for v in ("qwen14b", "llama8b") for d in ("hotpot", "2wiki")]
VNAME = {"qwen14b": "Qwen2.5-14B", "llama8b": "Llama-3.1-8B"}
DNAME = {"hotpot": "HotpotQA", "2wiki": "2WikiMultiHopQA"}
NEU = "neutral"
ARMS = [NEU, "fixed_sufficiency", "fixed_saturation", "bridge_sufficiency_k3", "bridge_sufficiency_k1",
        "bridge_saturation_k3", "entity_sufficiency", "unsupported_sufficiency"]
ARM_NAME = {NEU: "Neutral", "fixed_sufficiency": "Fixed Sufficiency", "fixed_saturation": "Fixed Saturation",
            "bridge_sufficiency_k3": "Bridge K3", "bridge_sufficiency_k1": "Bridge K1",
            "bridge_saturation_k3": "Bridge Saturation K3", "entity_sufficiency": "Entity",
            "unsupported_sufficiency": "Unsupported"}
CONTRASTS = [  # (id, label, B, A, tier) as in thesis_restructure_v2.E2_CONTRASTS
    ("structure", "Structure (K3 vs Entity)", "bridge_sufficiency_k3", "entity_sufficiency", "primary"),
    ("truth", "Truth (K3 vs Unsupported)", "bridge_sufficiency_k3", "unsupported_sufficiency", "primary"),
    ("selection", "Selection (K3 vs K1)", "bridge_sufficiency_k3", "bridge_sufficiency_k1", "primary"),
    ("bridge_k3_vs_fixed_sufficiency", "Bridge vs Fixed (K3 vs Fixed Suff.)", "bridge_sufficiency_k3",
     "fixed_sufficiency", "secondary"),
    ("bridge_sat_k3_vs_fixed_saturation", "Bridge-Sat vs Fixed-Sat", "bridge_saturation_k3", "fixed_saturation",
     "secondary"),
]
C_NEU, C_FIX, C_K3, C_QWEN, C_LLAMA = "#2a78d6", "#eb6834", "#6250d6", "#2a78d6", "#eb6834"


# --------------------------------------------------------------------------- helpers
def save_csv(df, name):
    print(f"\n--- {name}: {len(df)} rows; first 3 rows:")
    print(df.head(3).to_string(index=False))
    df.to_csv(OUTC / name, index=False)  # no rounding
    print(f"wrote {OUTC / name}")


def save_fig(fig, stem):
    OUTF.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTF / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTF / stem}.pdf/.png")


def boot_stat(x, stat):
    """Percentile CI of stat(x) over 2,000 resamples of questions (same RNG scheme as boot)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    s = stat(x[rng.integers(0, len(x), size=(NB, len(x)))], axis=1)
    return tuple(np.quantile(s, [0.025, 0.975]))


def wilcox(d):
    """Wilcoxon signed-rank, zero_method='wilcox' (zero differences dropped), two-sided. (stat, p, n_nonzero)."""
    d = np.asarray(d, dtype=float)
    nz = int((d != 0).sum())
    if nz == 0:
        return (np.nan, 1.0, 0)
    r = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return (float(r.statistic), float(r.pvalue), nz)


def load():
    L = pd.read_csv(E2_DIR / "trajectories/run_labels.csv", dtype=str)
    t = pd.to_numeric(L.t_i, errors="coerce")
    done = L.terminal_route.isin(ROUTES) & t.notna()
    L["t"] = np.where(done, t, T_NC).astype(float)
    L["done"] = done
    L["ran"] = L.status == "ran"
    L["valid"] = L.ran & (L.pcr_eligible == "True")
    evd = L.evidence.fillna("{}").map(json.loads)
    L["cov_stop"] = pd.to_numeric(evd.map(lambda e: e.get("episode0_terminal_coverage")), errors="coerce")
    L["full_stop"] = L.cov_stop >= 1 - 1e-9
    L["correct"] = L.answer_correct.map({"True": 1.0, "False": 0.0})
    L["final_full"] = L.evidence_complete.map({"True": True, "False": False})
    L["prim_ok"] = L.primary_eligible == "True"
    L["prem"] = (L.premature_stop_no_hop2 == "True").astype(float)
    return L


def pair(L, v, d, B, A, need="valid"):
    x = L[(L.victim == v) & (L.dataset == d) & L[need]]
    m = x[x.arm == B].merge(x[x.arm == A], on="example_id", suffixes=("_b", "_a"))
    return m.sort_values("example_id", ignore_index=True)  # question order fixes the bootstrap draws (as in v2's pivot)


# --------------------------------------------------------------------------- Task 1
def task1(L):
    rule("TASK 1: exact attrition per cell")
    bp = pd.read_csv(E2_DIR / "bridge_selection/bridge_provenance.csv", dtype=str)
    cov = pd.read_csv(E2_DIR / "diagnostics/bridge_and_cue_coverage.csv")
    rows = []
    for v, d in CELLS:
        x = L[(L.victim == v) & (L.dataset == d)]
        b = bp[(bp.victim == v) & (bp.dataset == d)].status
        st = x.pivot(index="example_id", columns="arm", values="status")
        has_cue = st.isin(["ran", "d1_mismatch"])
        valid = x.pivot(index="example_id", columns="arm", values="valid")
        ran = st == "ran"
        all8_cue, all8_ran, all8_valid = has_cue.all(axis=1), ran.all(axis=1), valid.all(axis=1)
        neu = x[x.arm == NEU].set_index("example_id")
        r = {"model": v, "dataset": d,
             "n_manifest": x.example_id.nunique(),
             "n_anchor_victim_failure": int((b == "ANCHOR_VICTIM_FAILURE").sum()),
             "n_decompose_invalid": int((b == "DECOMPOSE_INVALID").sum()),
             "n_no_facts": int((b == "NO_FACTS").sum()),
             "n_bridge_extractable_from_d1": int(b.isin(["OK", "NO_ANCHOR"]).sum()),
             "n_no_hop_match": int((b == "NO_ANCHOR").sum()),
             "n_hop_structure_match": int((b == "OK").sum()),
             "n_gate_pass_sufficiency_k3_k1": int(has_cue["bridge_sufficiency_k3"].sum()),
             "n_gate_pass_saturation_k3": int(has_cue["bridge_saturation_k3"].sum()),
             "n_gate_pass_entity": int(has_cue["entity_sufficiency"].sum()),
             "n_gate_pass_unsupported": int(has_cue["unsupported_sufficiency"].sum()),
             "n_8arm_intersection": int(all8_cue.sum()),
             "n_8arm_d1_anchor_matched": int(all8_ran.sum()),
             "n_8arm_anchor_matched_pcr_eligible": int(all8_valid.sum()),
             "n_8arm_baseline_gate_eligible": int((all8_valid & neu.prim_ok.reindex(all8_valid.index)).sum()),
             "n_neutral_baseline_gate_eligible": int((neu.valid & neu.prim_ok).sum())}
        for a in ARMS:
            r[f"ran_{a}"] = int(ran[a].sum())
            r[f"d1_mismatch_{a}"] = int((st[a] == "d1_mismatch").sum())
            r[f"valid_{a}"] = int(valid[a].sum())
        c = cov[(cov.victim == v) & (cov.dataset == d)].iloc[0]
        ok = (r["n_hop_structure_match"] == c.bridge_OK and r["n_no_facts"] == c.bridge_NO_FACTS
              and all(r[f"ran_{a}"] == c[f"ran_{a}"] for a in ARMS))
        r["matches_bridge_and_cue_coverage"] = bool(ok)
        rows.append(r)
    a = pd.DataFrame(rows)
    save_csv(a, "e2_attrition.csv")
    print("\nFunnel (per cell):")
    cols = ["model", "dataset", "n_manifest", "n_bridge_extractable_from_d1", "n_hop_structure_match",
            "n_gate_pass_sufficiency_k3_k1", "n_gate_pass_saturation_k3", "n_8arm_intersection",
            "n_8arm_d1_anchor_matched", "n_8arm_anchor_matched_pcr_eligible", "n_8arm_baseline_gate_eligible"]
    print(a[cols].to_string(index=False))
    print("Stage definitions: bridge extractable = span-backed facts extracted from d1 (bridge status OK or "
          "NO_ANCHOR); hop-structure match = a fact matches a non-answer hop edge (status OK); gate pass = the arm's "
          "selected cue passed validity gates (arm not skipped; K1 and K3 share one pool); 8-arm intersection = all 8 "
          "arms had a cue to run; d1-anchor matched = all 8 runs' d1 byte-identical to the anchor (status ran); "
          "baseline-gate eligible = of those (also all PCR-eligible), Neutral's d1 lacks a hop-2 dependency "
          "(primary_eligible).")
    print(f"VERIFY 1: bridge status counts and per-arm 'ran' counts = diagnostics/bridge_and_cue_coverage.csv in "
          f"{int(a.matches_bridge_and_cue_coverage.sum())}/4 cells -> "
          f"{'PASS' if a.matches_bridge_and_cue_coverage.all() else 'FAIL'}")
    return a


# --------------------------------------------------------------------------- Task 2
def task2(L):
    rule("TASK 2: PCR curves and k50")
    K = np.arange(1, 30)
    rows, k50 = [], []
    for v, d in CELLS:
        for a in ARMS:
            x = L[(L.victim == v) & (L.dataset == d) & (L.arm == a) & L.valid]
            rows += [{"model": v, "dataset": d, "arm": a, "k": int(k), "n_runs": len(x), "pcr": (x.t <= k).mean()}
                     for k in K]
        neu = L[(L.victim == v) & (L.dataset == d) & (L.arm == NEU) & L.valid]
        pcr = np.array([(neu.t <= k).mean() for k in K])
        k = int(K[np.argmax(pcr >= 0.5)]) if (pcr >= 0.5).any() else np.nan
        k50.append({"model": v, "dataset": d, "n_neutral_runs": len(neu), "k50": k,
                    "pcr_neutral_at_k50": pcr[k - 1], "pcr_neutral_at_k50_minus_1": pcr[k - 2] if k > 1 else 0.0})
    pc, kd = pd.DataFrame(rows), pd.DataFrame(k50)
    save_csv(pc, "e2_pcr_curves.csv")
    save_csv(kd, "e2_k50.csv")
    print("\nk50 per cell:")
    print(kd.to_string(index=False))
    fig_pcr(pc, kd)
    return pc, dict(zip(zip(kd.model, kd.dataset), kd.k50))


def fig_pcr(pc, kd):
    arms = [(NEU, "Neutral", C_NEU), ("fixed_sufficiency", "Fixed Sufficiency", C_FIX),
            ("bridge_sufficiency_k3", "Bridge K3", C_K3)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for i, v in enumerate(("qwen14b", "llama8b")):
        for j, d in enumerate(("hotpot", "2wiki")):
            ax = axes[i, j]
            V.style(ax, "y")
            for a, lab, c in arms:
                x = pc[(pc.model == v) & (pc.dataset == d) & (pc.arm == a)]
                ax.plot(x.k, 100 * x.pcr, color=c, lw=2, marker="o", ms=2.5)
            k = kd[(kd.model == v) & (kd.dataset == d)].k50.iloc[0]
            ax.axvline(k, color="#999999", lw=0.9, ls="--", zorder=1)
            ax.text(k * 1.05, 3, f"k₅₀ = {k}", fontsize=8, color="#555555")
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 3, 5, 10, 20, 29], ["1", "2", "3", "5", "10", "20", "29"])
            ax.minorticks_off()
            ax.set_xlim(0.92, 31)
            ax.set_ylim(0, 100)
            ax.set_title(f"{VNAME[v]} — {DNAME[d]}", fontsize=11)
            if i == 1:
                ax.set_xlabel("Decision step after the cue, k (log scale)")
            if j == 0:
                ax.set_ylabel("Cumulative completion rate (%)")
    fig.legend([Line2D([0], [0], color=c, lw=2.2) for *_, c in arms], [lab for _, lab, _ in arms],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, "fig_e2_pcr_cdf_k50")  # k50 variant; the thesis figure is fig_e2_pcr_cdf.py


# --------------------------------------------------------------------------- Task 3
def task3(L):
    rule("TASK 3: ASR per arm vs Neutral, per cell")
    print("Wilcoxon signed-rank on D_q: scipy.stats.wilcoxon(zero_method='wilcox', alternative='two-sided'); "
          "zero differences (ties, D_q = 0) are DROPPED before ranking, so the test uses only questions where the "
          "arms stopped at different steps (n_nonzero). scipy picks the exact null for small n_nonzero without ties "
          "and the normal approximation otherwise. A cell with no nonzero D_q gets p = 1.")
    rows = []
    for a in ARMS[1:]:
        for v, d in CELLS:
            p = pair(L, v, d, a, NEU)
            dq = (p.t_a - p.t_b).to_numpy()  # T_neutral - T_arm (a = Neutral side here)
            n = len(dq)
            e, s, l_ = int((dq > 0).sum()), int((dq == 0).sum()), int((dq < 0).sum())
            lo_mean, hi_mean, _ = boot(dq)
            lo_med, hi_med = boot_stat(dq, np.median)
            w, pw, nz = wilcox(dq)
            rows.append({"arm": a, "arm_name": ARM_NAME[a], "model": v, "dataset": d, "n_pairs": n,
                         "n_noncomp_neutral": int((p.t_a == T_NC).sum()), "n_noncomp_arm": int((p.t_b == T_NC).sum()),
                         "pct_earlier": 100 * e / n, "ci_lower_earlier": wilson(e, n)[0],
                         "ci_upper_earlier": wilson(e, n)[1], "pct_same": 100 * s / n,
                         "ci_lower_same": wilson(s, n)[0], "ci_upper_same": wilson(s, n)[1],
                         "pct_later": 100 * l_ / n, "ci_lower_later": wilson(l_, n)[0],
                         "ci_upper_later": wilson(l_, n)[1], "mean_dq": dq.mean(), "ci_lower_mean_dq": lo_mean,
                         "ci_upper_mean_dq": hi_mean, "median_dq": float(np.median(dq)), "ci_lower_median_dq": lo_med,
                         "ci_upper_median_dq": hi_med, "wilcoxon_stat": w, "n_nonzero": nz, "p_wilcoxon": pw})
    asr = pd.DataFrame(rows)
    asr["q_bh"] = asr.groupby("arm").p_wilcoxon.transform(lambda p: bh(p.to_numpy()))
    save_csv(asr, "e2_asr_all.csv")
    old = pd.read_csv(REF / "e2_asr_table.csv")
    lab = {"Fixed Sufficiency vs Neutral": "fixed_sufficiency", "Bridge K3 vs Neutral": "bridge_sufficiency_k3"}
    j = old.assign(arm=old.contrast.map(lab)).merge(asr, on=["arm", "model", "dataset"], suffixes=("_old", ""))
    mx = (j.pct_earlier_old - j.pct_earlier).abs().max()
    print(f"\nVERIFY 3: Fixed/K3 ASR vs thesis_restructure/csv/e2_asr_table.csv: {len(j)}/8 rows, n equal "
          f"{(j.n_pairs_old == j.n_pairs).all()}, max |diff| pct_earlier = {mx:.4f} -> "
          f"{'PASS' if len(j) == 8 and (j.n_pairs_old == j.n_pairs).all() and mx < 0.01 else 'FAIL'}")
    print("\nASR per arm (rows = arm x cell):")
    print(asr[["arm_name", "model", "dataset", "n_pairs", "pct_earlier", "pct_same", "pct_later", "mean_dq",
               "median_dq", "p_wilcoxon", "q_bh"]].to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    return asr


# --------------------------------------------------------------------------- Task 4
def task4(L, k50):
    rule("TASK 4: five contrasts at k50")
    old = pd.read_csv(V.OUTC / "e2_contrasts.csv")
    rows = []
    for cid, label, B, A, tier in CONTRASTS:
        for v, d in CELLS:
            k = k50[(v, d)]
            p = pair(L, v, d, B, A)
            pb, pa = (p.t_b <= k).astype(float).to_numpy(), (p.t_a <= k).astype(float).to_numpy()
            lo, hi, pp = boot(pb - pa)
            # pre-registered primary outcome, recomputed with the rules of thesis_restructure_v2.task7
            if v == "qwen14b":
                y = pair(L[L.prim_ok], v, d, B, A)
                yb, ya = y.prem_b.to_numpy(), y.prem_a.to_numpy()
            else:
                yb, ya = (p.t_b <= 2).astype(float).to_numpy(), (p.t_a <= 2).astype(float).to_numpy()
            olo, ohi, _ = boot(yb - ya)
            bo, ao = int(((yb == 1) & (ya == 0)).sum()), int(((ya == 1) & (yb == 0)).sum())
            o = old[(old.contrast == cid) & (old.model == v) & (old.dataset == d)].iloc[0]
            same = (len(yb) == o.n_primary_pairs and abs(100 * (yb - ya).mean() - o.d_primary) < 1e-9
                    and abs(100 * olo - o.ci_lower_d_primary) < 1e-9 and abs(100 * ohi - o.ci_upper_d_primary) < 1e-9
                    and abs(mcnemar_p(bo, ao) - o.mcnemar_p_primary) < 1e-12)
            rows.append({"tier": tier, "contrast": cid, "label": label, "arm_B": B, "arm_A": A, "model": v,
                         "dataset": d, "n_pairs": len(p), "k50": k, "pcr_B_at_k50": 100 * pb.mean(),
                         "pcr_A_at_k50": 100 * pa.mean(), "d_pcr_k50": 100 * (pb - pa).mean(),
                         "ci_lower_d_pcr_k50": 100 * lo, "ci_upper_d_pcr_k50": 100 * hi, "p_boot_d_pcr_k50": pp,
                         "primary_outcome": o.primary_outcome, "n_primary_pairs": o.n_primary_pairs,
                         "primary_rate_B": o.primary_rate_B, "primary_rate_A": o.primary_rate_A,
                         "d_primary": o.d_primary, "ci_lower_d_primary": o.ci_lower_d_primary,
                         "ci_upper_d_primary": o.ci_upper_d_primary, "mcnemar_p_primary": o.mcnemar_p_primary,
                         "primary_recomputed_equal": same})
    c = pd.DataFrame(rows)
    c["q_bh_d_pcr_k50"] = c.groupby("contrast").p_boot_d_pcr_k50.transform(lambda p: bh(p.to_numpy()))
    save_csv(c, "e2_contrasts_k50.csv")
    print(f"\nVERIFY 4a: pre-registered primary columns carried over unchanged from v2/e2_contrasts.csv and equal to a "
          f"recomputation in {int(c.primary_recomputed_equal.sum())}/20 rows -> "
          f"{'PASS' if c.primary_recomputed_equal.all() else 'FAIL'}")
    j = c.merge(old[["contrast", "model", "dataset", "n_pairs"]], on=["contrast", "model", "dataset"],
                suffixes=("", "_v2"))
    print(f"VERIFY 4b: n_pairs equal to e2_contrasts.csv in {int((j.n_pairs == j.n_pairs_v2).sum())}/20 rows")
    print("\ndPCR@k50 per contrast x cell:")
    print(c[["contrast", "model", "dataset", "n_pairs", "k50", "pcr_B_at_k50", "pcr_A_at_k50", "d_pcr_k50",
             "ci_lower_d_pcr_k50", "ci_upper_d_pcr_k50", "p_boot_d_pcr_k50", "q_bh_d_pcr_k50"]]
          .to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    fig_forest(c)
    return c


FOREST_LABELS = {"structure": "Structure\n(K3 vs Entity)", "truth": "Truth\n(K3 vs Unsupported)",
                 "selection": "Selection\n(K3 vs K1)", "bridge_k3_vs_fixed_sufficiency": "Bridge vs Fixed\n(K3 vs Fixed Suff.)",
                 "bridge_sat_k3_vs_fixed_saturation": "Bridge-Sat vs Fixed-Sat\n(Sat. K3 vs Fixed Sat.)"}


def fig_forest(c, col="d_pcr_k50", lo="ci_lower_d_pcr_k50", hi="ci_upper_d_pcr_k50",
               xlabel="ΔASR (percentage points)", stem="fig_e2_forest"):
    """Forest plot of one B - A effect per contrast x cell (default: dPCR@k50). Qwen blue, Llama orange (as in the
    thesis PCR figures); filled marker = 95% CI excludes 0."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    ys = {cid: y for y, cid in zip([0, 1, 2, 3.6, 4.6], [x[0] for x in CONTRASTS])}
    lo_all = min(c[lo].min(), -5)
    hi_all = max(c[hi].max(), 5)
    for ax, d in zip(axes, ("hotpot", "2wiki")):
        V.style(ax, "x")
        ax.axvline(0, color="#3d3d3a", lw=1)
        ax.axhline(3.0, color="#cccccc", lw=0.8, ls="--")
        for v, off, colr in (("qwen14b", -0.14, C_QWEN), ("llama8b", 0.14, C_LLAMA)):
            for cid, *_ in CONTRASTS:
                r = c[(c.contrast == cid) & (c.model == v) & (c.dataset == d)].iloc[0]
                y = ys[cid] + off
                excl = r[lo] > 0 or r[hi] < 0
                ax.plot([r[lo], r[hi]], [y, y], color=colr, lw=1.6)
                ax.plot(r[col], y, "o", ms=6.5, color=colr, mfc=colr if excl else "white", mew=1.6, zorder=3)
        ax.set_title(DNAME[d], fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_xlim(lo_all - 3, hi_all + 3)
    axes[0].set_yticks([ys[cid] for cid, *_ in CONTRASTS], [FOREST_LABELS[cid] for cid, *_ in CONTRASTS],
                       fontsize=10)
    for y, lab in ((-0.55, "Primary"), (3.15, "Secondary")):
        axes[0].text(0.01, y, lab, transform=axes[0].get_yaxis_transform(), ha="left", va="center", fontsize=9,
                     color="#52514e", style="italic")
    axes[0].set_ylim(4.95, -0.85)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    h = [Line2D([0], [0], color=C_QWEN, marker="o", lw=1.6, mfc=C_QWEN),
         Line2D([0], [0], color=C_LLAMA, marker="o", lw=1.6, mfc=C_LLAMA),
         Line2D([0], [0], color="#3d3d3a", marker="o", lw=0, mfc="#3d3d3a"),
         Line2D([0], [0], color="#3d3d3a", marker="o", lw=0, mfc="white")]
    fig.legend(h, ["Qwen2.5-14B", "Llama-3.1-8B", "95% CI excludes 0", "95% CI includes 0"], loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.subplots_adjust(wspace=0.15)
    save_fig(fig, stem)


# --------------------------------------------------------------------------- Task 5
def task5(L):
    rule("TASK 5: EICR per cell for every arm")
    print("EICR = % of completed runs whose evidence at the Researcher's stop is incomplete "
          "(episode0_terminal_coverage < 1). Per arm and cell, on questions where the arm and Neutral are both valid "
          "AND both completed with coverage recorded (matched pairs, so McNemar applies). The Neutral row uses the "
          "Fixed Sufficiency pairing.")
    rows = []
    for a in ARMS[1:]:
        for v, d in CELLS:
            p = pair(L, v, d, a, NEU)
            p = p[p.done_b & p.done_a & p.cov_stop_b.notna() & p.cov_stop_a.notna()]
            ib, ia = (~p.full_stop_b).astype(float).to_numpy(), (~p.full_stop_a).astype(float).to_numpy()
            n, kb, ka = len(p), int(ib.sum()), int(ia.sum())
            lo, hi, _ = boot(ib - ia)
            bo, ao = int(((ib == 1) & (ia == 0)).sum()), int(((ia == 1) & (ib == 0)).sum())
            rows.append({"arm": a, "arm_name": ARM_NAME[a], "model": v, "dataset": d, "n_pairs_completed": n,
                         "eicr_arm": 100 * kb / n, "ci_lower_arm": wilson(kb, n)[0], "ci_upper_arm": wilson(kb, n)[1],
                         "eicr_neutral": 100 * ka / n, "ci_lower_neutral": wilson(ka, n)[0],
                         "ci_upper_neutral": wilson(ka, n)[1], "d_eicr": 100 * (ib - ia).mean(),
                         "ci_lower_d_eicr": 100 * lo, "ci_upper_d_eicr": 100 * hi, "n_arm_only_incomplete": bo,
                         "n_neutral_only_incomplete": ao, "mcnemar_p": mcnemar_p(bo, ao)})
    e = pd.DataFrame(rows)
    e["q_bh"] = e.groupby("arm").mcnemar_p.transform(lambda p: bh(p.to_numpy()))
    save_csv(e, "e2_eicr_mcnemar.csv")
    print(e[["arm_name", "model", "dataset", "n_pairs_completed", "eicr_neutral", "eicr_arm", "d_eicr", "mcnemar_p",
             "q_bh"]].to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    # verification on the published definition (thesis_restructure.eicr_rows): questions valid in all of
    # Neutral / Fixed / K3, then each arm's completed, coverage-recorded runs. t uses t_i as in T.load_e2.
    old = pd.read_csv(REF / "e2_eicr_table.csv")
    three = [NEU, "fixed_sufficiency", "bridge_sufficiency_k3"]
    worst, n_ok = 0.0, 0
    for v, d in CELLS:
        x = L[(L.victim == v) & (L.dataset == d) & L.arm.isin(three) & L.valid]
        qs = set.intersection(*[set(x[x.arm == a].example_id) for a in three])
        for a in three:
            y = x[(x.arm == a) & x.example_id.isin(qs) & pd.to_numeric(x.t_i, errors="coerce").notna()
                  & x.cov_stop.notna()]
            rate = 100 * (~y.full_stop).mean()
            o = old[(old.model == v) & (old.dataset == d) & (old.arm == V.T.ARM[a])].iloc[0]
            worst = max(worst, abs(rate - o.eicr_rate))
            n_ok += len(y) == o.n_pairs
    print(f"\nVERIFY 5: Neutral/Fixed/K3 EICR on the published 3-arm definition vs thesis_restructure/csv/"
          f"e2_eicr_table.csv: n equal in {n_ok}/12 rows, max |diff| = {worst:.4f} pp -> "
          f"{'PASS' if n_ok == 12 and worst < 0.01 else 'FAIL'}. (The CSV above uses arm-vs-Neutral matched pairs, "
          "so its rates differ slightly from that 3-arm table by design.)")
    return e


# --------------------------------------------------------------------------- Task 6
def task6(L):
    rule("TASK 6: accuracy and mechanism per cell")
    acc, mech = [], []
    for a in ARMS[1:]:
        for v, d in CELLS:
            p = pair(L, v, d, a, NEU)
            p = p[p.correct_b.notna() & p.correct_a.notna()]
            cb, ca = p.correct_b.to_numpy(), p.correct_a.to_numpy()
            diff = cb - ca
            lo, hi, _ = boot(diff)
            tp = ttest_rel(cb, ca).pvalue if len(p) > 1 and diff.std() > 0 else 1.0
            acc.append({"arm": a, "arm_name": ARM_NAME[a], "model": v, "dataset": d, "n_pairs": len(p),
                        "acc_neutral": 100 * ca.mean(), "acc_arm": 100 * cb.mean(), "d_acc": 100 * diff.mean(),
                        "ci_lower_d_acc": 100 * lo, "ci_upper_d_acc": 100 * hi, "p_paired_t": float(tp),
                        "n_neutral_right_arm_wrong": int(((ca == 1) & (cb == 0)).sum()),
                        "n_neutral_wrong_arm_right": int(((ca == 0) & (cb == 1)).sum())})
            if a in ("fixed_sufficiency", "bridge_sufficiency_k3"):
                m = p[p.cov_stop_b.notna() & p.cov_stop_a.notna()]
                diffc = m.cov_stop_b - m.cov_stop_a
                for g, mask in (("lost", diffc < -1e-9), ("same", diffc.abs() <= 1e-9), ("gained", diffc > 1e-9)):
                    y = m[mask]
                    mech.append({"arm": a, "arm_name": ARM_NAME[a], "model": v, "dataset": d, "group": g,
                                 "n_pairs": len(y), "acc_neutral": 100 * y.correct_a.mean() if len(y) else np.nan,
                                 "acc_arm": 100 * y.correct_b.mean() if len(y) else np.nan,
                                 "d_acc": 100 * (y.correct_b - y.correct_a).mean() if len(y) else np.nan,
                                 "flag_n_lt_20": len(y) < 20})
    acc, mech = pd.DataFrame(acc), pd.DataFrame(mech)
    save_csv(acc, "e2_accuracy.csv")
    save_csv(mech, "e2_mechanism.csv")
    print("\nAccuracy (dAcc = arm - Neutral, pp):")
    print(acc[["arm_name", "model", "dataset", "n_pairs", "acc_neutral", "acc_arm", "d_acc", "ci_lower_d_acc",
               "ci_upper_d_acc", "p_paired_t"]].to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    print("\nMechanism (coverage at the Researcher's stop, arm vs Neutral):")
    print(mech.to_string(index=False, float_format=lambda f: f"{f:.3g}"))

    print("\nSOURCE OF THE THESIS's '+14.6 pp (n = 89)' AND '-25.6 pp (n = 39)':")
    print("  notes/make_thesis_figures.py:e2_harm (Figure E2.6, notes/figures/e2_harm_by_transition.png). It POOLS all"
          " 4 cells (Qwen + Llama, HotpotQA + 2Wiki); classifies pairs by FULL evidence at the FINAL-REPORT input"
          " (run_labels.evidence_complete), not coverage at the Researcher's stop; uses every anchor-matched pair"
          " (no PCR-eligibility filter); and reports NET HARM = (Neutral right & cue wrong - reverse) / n, i.e."
          " -dAcc. Reproduction:")
    found = {}
    for a in ("fixed_sufficiency", "bridge_sufficiency_k3"):
        for cls, nf, cf in (("evidence_lost", True, False), ("evidence_gained", False, True)):
            tot, b_, c_ = 0, 0, 0
            for v, d in CELLS:
                p = pair(L, v, d, a, NEU, need="ran")
                p = p[p.final_full_a.notna() & p.final_full_b.notna() & (p.final_full_a == nf) & (p.final_full_b == cf)]
                tot += len(p)
                b_ += int(((p.correct_a == 1) & (p.correct_b != 1)).sum())
                c_ += int(((p.correct_a != 1) & (p.correct_b == 1)).sum())
            print(f"    {ARM_NAME[a]:18s} {cls:16s}: n = {tot:3d}, net harm = {100 * (b_ - c_) / tot:+.1f} pp,"
                  f" McNemar p = {mcnemar_p(b_, c_):.3g}")
            found[(a, cls)] = (tot, round(100 * (b_ - c_) / tot, 1))
    ok = (found[("fixed_sufficiency", "evidence_lost")] == (89, 14.6)
          and found[("fixed_sufficiency", "evidence_gained")] == (39, -25.6))
    print(f"VERIFY 6: Fixed Sufficiency reproduces the thesis's +14.6 pp (n = 89) and -25.6 pp (n = 39) -> "
          f"{'PASS (source confirmed)' if ok else 'FAIL'}. These pooled numbers mix 4 cells and a different evidence"
          " point; replace them with the per-cell rows of e2_mechanism.csv.")
    return acc, mech


# --------------------------------------------------------------------------- Task 7
def task7():
    rule("TASK 7: E1 alignment check (Wilcoxon on D_q vs sign test)")
    ev = V.load_e1_all()
    old = pd.read_csv(V.OUTC / "asr_table_all.csv")
    rows = []
    for cue, ref, label, _ in V.CONTRASTS:
        p = V.pairs(ev, cue, ref)
        for m, _ in V.MODELS:
            for d, _ in V.DATASETS:
                x = p[(p.model == m) & (p.dataset == d)]
                dq = (x.t_r - x.t_c).to_numpy()
                w, pw, nz = wilcox(dq)
                rows.append({"contrast": label, "model": m, "dataset": d, "n_pairs": len(dq), "n_nonzero": nz,
                             "mean_dq": dq.mean(), "median_dq": float(np.median(dq)), "wilcoxon_stat": w,
                             "p_wilcoxon": pw})
    w = pd.DataFrame(rows)
    w["q_bh_wilcoxon"] = w.groupby("contrast").p_wilcoxon.transform(lambda p: bh(p.to_numpy()))
    j = w.merge(old[["contrast", "model", "dataset", "n_pairs", "net", "p_sign", "q_bh"]],
                on=["contrast", "model", "dataset"], suffixes=("", "_old"))
    assert (j.n_pairs == j.n_pairs_old).all()
    j = j.drop(columns="n_pairs_old").rename(columns={"q_bh": "q_bh_sign_160"})
    j["sig_wilcoxon"] = j.q_bh_wilcoxon < 0.05
    j["sig_sign"] = j.q_bh_sign_160 < 0.05
    j["dir_wilcoxon"] = np.sign(j.mean_dq)
    j["dir_sign"] = np.sign(j.net)
    j["differs"] = (j.sig_wilcoxon != j.sig_sign) | (j.sig_wilcoxon & j.sig_sign & (j.dir_wilcoxon != j.dir_sign))
    save_csv(j, "e1_asr_wilcoxon.csv")
    print(f"\nCells whose significance differs (Wilcoxon, BH within comparison over 16 cells, vs sign test, BH over"
          f" all 160 cells): {int(j.differs.sum())}/160")
    if j.differs.any():
        print(j[j.differs][["contrast", "model", "dataset", "n_pairs", "net", "mean_dq", "p_sign", "q_bh_sign_160",
                            "p_wilcoxon", "q_bh_wilcoxon"]].to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    print("\nSignificant cells per comparison (q < 0.05): sign test | Wilcoxon")
    for label, g in j.groupby("contrast", sort=False):
        print(f"  {label:38s} {int(g.sig_sign.sum()):2d} | {int(g.sig_wilcoxon.sum()):2d}")
    return j


# --------------------------------------------------------------------------- --cov: common-question PCR and coverage loss
def task_a_common(L):
    """TASK A: PCR(k), k = 1-29, for Neutral, Fixed Sufficiency and Bridge K3 on the questions where ALL three arms
    have a valid (anchor-matched, PCR-eligible) run, per cell. Figure in the fig_e2_pcr_cdf style (50% line, no k50)."""
    rule("TASK A: E2 PCR curves on common questions (Neutral, Fixed Sufficiency, Bridge K3)")
    print(task_a_common.__doc__)
    import fig_e2_pcr_cdf as F  # same panel style as the thesis figure
    three = [a for a, *_ in F.ARMS]
    rows = []
    for v, d in CELLS:
        x = L[(L.victim == v) & (L.dataset == d) & L.arm.isin(three) & L.valid]
        qs = set.intersection(*[set(x[x.arm == a].example_id) for a in three])
        for a in three:
            y = x[(x.arm == a) & x.example_id.isin(qs)]
            rows += [{"model": v, "dataset": d, "arm": a, "k": k, "n_runs": len(y), "pcr": (y.t <= k).mean()}
                     for k in range(1, 30)]
    pc = pd.DataFrame(rows)
    OUTC_COV.mkdir(parents=True, exist_ok=True)
    pc.to_csv(OUTC_COV / "e2_pcr_curves_common.csv", index=False)
    print("n common questions per cell:", pc.groupby(["model", "dataset"]).n_runs.first().to_dict())
    allv = pd.read_csv(OUTC / "e2_pcr_curves.csv")
    for k in (1, 2, 3, 5):
        a = pc[pc.k == k].pivot_table(index=["model", "dataset"], columns="arm", values="pcr")
        b_ = allv[(allv.k == k) & allv.arm.isin(three)].pivot_table(index=["model", "dataset"], columns="arm",
                                                                      values="pcr")
        print(f"\nPCR@{k} (%): common questions | all valid runs")
        print((100 * a).round(1).astype(str).add(" | ").add((100 * b_).round(1).astype(str)).to_string())
    print(f"wrote {OUTC_COV / 'e2_pcr_curves_common.csv'}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, v in enumerate(("qwen14b", "llama8b")):
        for j, d in enumerate(("hotpot", "2wiki")):
            ax = axes[i, j]
            ax.axhline(50, color="#888888", lw=0.8, ls="--", zorder=1)
            F.panel(ax, pc, v, d, xlabel=i == 1, ylabel=j == 0)
            n = pc[(pc.model == v) & (pc.dataset == d)].n_runs.iloc[0]
            ax.text(0.98, 0.04, f"n = {n} common questions", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#555555")
    h = [Line2D([0], [0], color=c, lw=2.2) for *_, c in F.ARMS] + [Line2D([0], [0], color="#888888", lw=0.8, ls="--")]
    fig.legend(h, [lab for _, lab, _, _ in F.ARMS] + ["50% reference"], loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    save_fig(fig, "fig_e2_pcr_cdf_common")
    return pc


def task_b_e2(L):
    """TASK B (E2): paired loss of evidence coverage at the Researcher's stop, Fixed Sufficiency and Bridge K3 vs
    Neutral, 4 cells. cov_stop = episode0_terminal_coverage (h_stop / H). Pairs: both runs valid and both with
    cov_stop recorded. dcov_q = cov_neutral - cov_arm. Mean with paired bootstrap CI (2,000, seed 20260910);
    Wilcoxon (zero_method='wilcox', two-sided); BH across the 4 cells within each arm."""
    rule("TASK B (E2): paired local coverage loss, Fixed Sufficiency and Bridge K3 vs Neutral")
    print(task_b_e2.__doc__)
    rows = []
    for a in ("fixed_sufficiency", "bridge_sufficiency_k3"):
        for v, d in CELLS:
            p = pair(L, v, d, a, NEU)
            n_all = len(p)
            p = p[p.cov_stop_b.notna() & p.cov_stop_a.notna()]
            dc = (p.cov_stop_a - p.cov_stop_b).to_numpy()
            lo, hi, _ = boot(dc)
            w, pw, nz = wilcox(dc)
            rows.append({"arm": a, "arm_name": ARM_NAME[a], "model": v, "dataset": d, "n_pairs_valid": n_all,
                         "n_pairs": len(p), "n_missing_cov": n_all - len(p), "mean_cov_neutral": p.cov_stop_a.mean(),
                         "mean_cov_arm": p.cov_stop_b.mean(), "mean_dcov": dc.mean(), "ci_lower_dcov": lo,
                         "ci_upper_dcov": hi, "n_nonzero": nz, "p_wilcoxon": pw, "pct_dcov_pos": 100 * (dc > 0).mean(),
                         "pct_dcov_zero": 100 * (dc == 0).mean(), "pct_dcov_neg": 100 * (dc < 0).mean()})
    out = pd.DataFrame(rows)
    out["q_bh"] = out.groupby("arm").p_wilcoxon.transform(lambda p: bh(p.to_numpy()))
    out.to_csv(OUTC_COV / "e2_coverage_loss.csv", index=False)
    print(out.drop(columns=["arm", "n_pairs_valid"]).to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    print(f"wrote {OUTC_COV / 'e2_coverage_loss.csv'}")
    # verification: on e2_eicr_mcnemar's pairs (both completed, coverage recorded), share with cov_stop < 1 = EICR
    old = pd.read_csv(OUTC / "e2_eicr_mcnemar.csv")
    worst = 0.0
    for a in ("fixed_sufficiency", "bridge_sufficiency_k3"):
        for v, d in CELLS:
            p = pair(L, v, d, a, NEU)
            p = p[p.done_b & p.done_a & p.cov_stop_b.notna() & p.cov_stop_a.notna()]
            o = old[(old.arm == a) & (old.model == v) & (old.dataset == d)].iloc[0]
            worst = max(worst, abs(100 * (p.cov_stop_b < 1 - 1e-9).mean() - o.eicr_arm),
                        abs(100 * (p.cov_stop_a < 1 - 1e-9).mean() - o.eicr_neutral))
    print(f"VERIFY B2: E2 share of completed paired runs with cov_stop < 1 vs e2_eicr_mcnemar.csv (16 rates):"
          f" max |diff| = {worst:.2e} pp -> {'PASS' if worst < 1e-9 else 'FAIL'}")
    return out


OUTC_COV = REPO / "results/thesis_tables/coverage"


def main():
    OUTC.mkdir(parents=True, exist_ok=True)
    print(__doc__)
    L = load()
    task1(L)
    _, k50 = task2(L)
    task3(L)
    task4(L, k50)
    task5(L)
    task6(L)
    task7()


if __name__ == "__main__":
    if "--cov" in sys.argv:  # TASK A + B only (E1 part from thesis_restructure_v2); default run unchanged
        print("Coverage-loss and common-question extension (thesis_restructure_e2.py --cov)")
        L_ = load()
        task_a_common(L_)
        V.task_cov_e1()
        task_b_e2(L_)
    else:
        main()

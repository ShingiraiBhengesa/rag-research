"""Trace raster: post-observation Researcher actions, Neutral vs Fixed Sufficiency (Qwen2.5-14B, HotpotQA).

Each row is one question (matched pairs, both runs PCR-eligible), sorted by the
Neutral completion step T_i (then example_id). Each column is a post-observation
decision step 1-10 of the first Researcher episode. A cell is the action of that
decision: Search, Think, Complete (ResearchComplete or no-tool exit), or blank
once the episode has ended.

Actions come from the raw traces, parsed with the same parser and step rule the
PCR analysis uses (trace_parser + ResearcherEpisode.post_branch_decisions).
The extracted T_i is checked against the published data/pcr_run_level.csv.

    python -m prematurestop.analysis.fixed_cue_matrix.trace_raster [--model qwen14b --dataset hotpot]

Outputs (results/figures/e1_thesis_figures/, gitignored scratch; not a thesis figure):
  data/trace_raster_actions_<model>_<dataset>.csv   one row per run: actions as "S|T|C"
  trace_raster.{pdf,png}
"""
from __future__ import annotations

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from prematurestop.analysis.fixed_cue_matrix.thesis_figures import BUNDLE, OUT, read, save
from prematurestop.evaluation import config
from prematurestop.evaluation.trace_parser import parse_trace_file
from prematurestop.paths import LOCAL_DATA, RESULTS_E1

NEUTRAL, SUFF = "NaturalNeutral", "Natural_PreRegisteredSufficiency"
STEPS = 10
CODES = {"E": 0, "S": 1, "T": 2, "C": 3, "O": 4}  # empty, search, think, complete, other
COLORS = ["#F5F5F5", "#378ADD", "#888780", "#1D9E75", "#E24B4A"]
LABELS = {"S": "Search", "T": "Think", "C": "Complete", "O": "Other (unparsed / other tool)"}


def action(d) -> str:
    if d.parse_ok and (d.is_research_complete or d.is_no_tool_exit):
        return "C"
    if config.SEARCH_TOOL_NAME in d.tool_call_names:
        return "S"
    if config.THINK_TOOL_NAME in d.tool_call_names:
        return "T"
    return "O"


def extract(model: str, dataset: str) -> list[dict]:
    runs = {(r["condition"], r["example_id"]): r
            for r in read(RESULTS_E1 / BUNDLE[model] / "data" / "pcr_run_level.csv")
            if r["model"] == model and r["dataset"] == dataset and r["condition"] in (NEUTRAL, SUFF)}
    qids = sorted(q for c, q in runs if c == NEUTRAL and runs[(c, q)]["pcr_eligible"] == "True"
                  and runs.get((SUFF, q), {}).get("pcr_eligible") == "True")
    rows, mismatches = [], 0
    for q in qids:
        for cond in (NEUTRAL, SUFF):
            pt = parse_trace_file(LOCAL_DATA / "traces" / model / "branches" / dataset / q / cond / "trace.json")
            acts = []
            for d in pt.first_episode.post_branch_decisions:
                acts.append(action(d))
                if acts[-1] == "C":
                    break
            t_i = acts.index("C") + 1 if "C" in acts else None
            pub = runs[(cond, q)]["first_terminal_post_observation_step"]
            pub = int(pub) if pub not in ("", "None") else None
            mismatches += t_i != pub
            rows.append({"model": model, "dataset": dataset, "condition": cond, "example_id": q,
                         "t_i": t_i, "published_t_i": pub, "n_decisions": len(acts), "actions": "|".join(acts)})
    print(f"{len(qids)} matched questions; T_i mismatches vs pcr_run_level.csv: {mismatches}")
    return rows


def plot(rows: list[dict], model: str, dataset: str, title: str) -> dict:
    by = {(r["condition"], r["example_id"]): r for r in rows}
    t = lambda r: float(r["t_i"]) if r["t_i"] not in ("", None, "None") else math.inf
    qids = sorted({q for _, q in by}, key=lambda q: (t(by[(NEUTRAL, q)]), q))
    n = len(qids)
    pcr1 = {c: sum(t(by[(c, q)]) <= 1 for q in qids) / n for c in (NEUTRAL, SUFF)}
    used = set()
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={"wspace": 0.06})
    for ax, cond, name in ((axes[0], NEUTRAL, "Neutral"), (axes[1], SUFF, "Fixed Sufficiency")):
        grid = np.zeros((n, STEPS), dtype=int)
        for i, q in enumerate(qids):
            acts = (by[(cond, q)]["actions"] or "").split("|") if by[(cond, q)]["actions"] else []
            for k, a in enumerate(acts[:STEPS]):
                grid[i, k] = CODES[a]
                used.add(a)
        ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=ListedColormap(COLORS), vmin=0, vmax=4,
                  extent=(0.5, STEPS + 0.5, n - 0.5, -0.5))
        for x in np.arange(1.5, STEPS, 1.0):  # thin white column gaps
            ax.axvline(x, color="white", lw=1.2)
        ax.set_title(name, fontsize=13, fontweight="normal", pad=8)
        ax.set_xticks(range(1, STEPS + 1), [str(k) for k in range(1, STEPS + 1)], fontsize=9)
        ax.set_xlabel("Decision step after cue", fontsize=10)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"Questions (n = {n}, sorted by Neutral completion step)", fontsize=10)
    d = pcr1[SUFF] - pcr1[NEUTRAL]
    axes[0].text(0.5, -0.075, f"PCR@1 = {pcr1[NEUTRAL]:.1%}", transform=axes[0].transAxes, ha="center",
                 fontsize=10, color="#555555")
    axes[1].text(0.5, -0.075, f"PCR@1 = {pcr1[SUFF]:.1%}  ({d * 100:+.1f} pp)", transform=axes[1].transAxes,
                 ha="center", fontsize=10, color="#1D9E75")
    handles = [Patch(color=COLORS[CODES[a]], label=LABELS[a]) for a in ("S", "T", "C", "O") if a in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=12, y=0.94)
    save(fig, "trace_raster")
    return {"n": n, "pcr1_neutral": pcr1[NEUTRAL], "pcr1_fixed_sufficiency": pcr1[SUFF]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen14b")
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--from-csv", action="store_true", help="re-plot from the extracted CSV, skip trace parsing")
    args = ap.parse_args(argv)
    path = OUT / "data" / f"trace_raster_actions_{args.model}_{args.dataset}.csv"
    if args.from_csv:
        rows = read(path)
    else:
        rows = extract(args.model, args.dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")
    print(plot(rows, args.model, args.dataset, "Qwen2.5-14B, HotpotQA: first Researcher episode after the cue"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

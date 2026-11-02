"""E1 Fixed Sufficiency vs Neutral, 16 model x dataset cells: EICR reconciliation (Pull 1) and timing reconciliation
(Pull 2).

Run from the repo root after thesis_restructure_v2.py:  PYTHONPATH=src python scripts/thesis_figures/eicr_timing_reconciled.py
Outputs: results/thesis_tables/e1/eicr_reconciled.csv, results/thesis_tables/e1/timing_reconciled.csv

Pull 1 (loader: thesis_restructure.load_e1)
  eligible   PCR-eligible run
  completed  eligible run that completed (RESEARCH_COMPLETE / NO_TOOL_DIRECT_EXIT, T < 30) AND has evidence at the
             Researcher's stop recorded (evidence_eligible, local_full_evidence True/False); completed runs without a
             recorded evidence value are counted in n_completed_no_evidence_* and left out of EICR, as in eicr_table.csv
  incomplete completed run with local_full_evidence False (coverage at the Researcher's stop < 1)
  eicr_*     unpaired, per arm; delta_eicr = eicr_cue - eicr_neutral (unpaired)
  pairs      same question, both eligible and both completed (as above); McNemar exact on the discordant pairs,
             BH over the 16 cells; delta CI = paired bootstrap of (cue incomplete - neutral incomplete) over pairs
             (2,000 resamples, seed 20260910), reported with the paired delta it brackets (delta_eicr_paired)
Pull 2 (loader: thesis_restructure_v2.load_e1_all, non-completion T = 30)
  D_q = T_neutral - T_cue; asr / reverse_rate over all eligible pairs; *_bothcompleted over pairs where both completed
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_restructure as T  # noqa: E402
import thesis_restructure_v2 as V  # noqa: E402

OUT = V.OUTC


def pull1():
    ev = T.load_e1()
    rows = []
    for m, _ in T.MODELS:
        for d, _ in T.DATASETS:
            x = ev[(ev.model == m) & (ev.dataset == d) & ev.pe]
            r = {"model": m, "dataset": d}
            for arm, sfx in ((T.NEU, "neutral"), (T.FIX, "cue")):
                y = x[x.condition == arm]
                done = y[np.isfinite(y.t)]
                comp = done[done.evok]
                k = int((~comp.local_full).sum())
                lo, hi = T.wilson(k, len(comp))
                r.update({f"n_eligible_{sfx}": len(y), f"n_completed_{sfx}": len(comp),
                          f"n_completed_no_evidence_{sfx}": len(done) - len(comp), f"n_incomplete_{sfx}": k,
                          f"eicr_{sfx}": 100 * k / len(comp), f"ci_lower_eicr_{sfx}": lo, f"ci_upper_eicr_{sfx}": hi})
            r["delta_eicr"] = r["eicr_cue"] - r["eicr_neutral"]
            n_ = x[(x.condition == T.NEU) & np.isfinite(x.t) & x.evok]
            c_ = x[(x.condition == T.FIX) & np.isfinite(x.t) & x.evok]
            p = c_.merge(n_, on="example_id", suffixes=("_c", "_n")).sort_values("example_id")
            ic, in_ = (~p.local_full_c).to_numpy(), (~p.local_full_n).to_numpy()
            b_, c2 = int((in_ & ~ic).sum()), int((ic & ~in_).sum())
            lo, hi, _ = V.boot(ic.astype(float) - in_.astype(float))
            r.update({"n_pairs": len(p), "n_pairs_neutral_incomplete": b_, "n_pairs_cue_incomplete": c2,
                      "n_pairs_both_incomplete": int((ic & in_).sum()),
                      "n_pairs_neither_incomplete": int((~ic & ~in_).sum()), "mcnemar_p": V.mcnemar_p(b_, c2),
                      "delta_eicr_paired": 100 * (ic.mean() - in_.mean()), "ci_lower_delta": 100 * lo,
                      "ci_upper_delta": 100 * hi})
            rows.append(r)
    out = pd.DataFrame(rows)
    out["mcnemar_q"] = V.bh(out.mcnemar_p.to_numpy())
    cols = ["model", "dataset", "n_eligible_neutral", "n_completed_neutral", "n_incomplete_neutral", "eicr_neutral",
            "n_eligible_cue", "n_completed_cue", "n_incomplete_cue", "eicr_cue", "delta_eicr", "n_pairs",
            "n_pairs_neutral_incomplete", "n_pairs_cue_incomplete", "n_pairs_both_incomplete",
            "n_pairs_neither_incomplete", "mcnemar_p", "mcnemar_q", "ci_lower_eicr_neutral", "ci_upper_eicr_neutral",
            "ci_lower_eicr_cue", "ci_upper_eicr_cue", "delta_eicr_paired", "ci_lower_delta", "ci_upper_delta",
            "n_completed_no_evidence_neutral", "n_completed_no_evidence_cue"]
    out = out[cols]
    out.to_csv(OUT / "eicr_reconciled.csv", index=False)
    print(out[["model", "dataset", "n_completed_neutral", "eicr_neutral", "n_completed_cue", "eicr_cue", "delta_eicr",
               "n_pairs", "delta_eicr_paired", "ci_lower_delta", "ci_upper_delta", "mcnemar_p", "mcnemar_q"]]
          .to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    old = pd.read_csv(OUT / "eicr_reconciliation.csv")
    old = old[old.dataset != "macro"]
    j = out.merge(old[old.arm == "Fixed Sufficiency"], on=["model", "dataset"], suffixes=("", "_v2"))
    jn = out.merge(old[old.arm == "Neutral"], on=["model", "dataset"], suffixes=("", "_v2"))
    ok = (len(j) == 16 and (j.eicr_cue - j.eicr_unpaired).abs().max() < 1e-9
          and (jn.eicr_neutral - jn.eicr_unpaired).abs().max() < 1e-9
          and (j.n_pairs == j.n_paired_both_completed).all() and (j.mcnemar_p - j.mcnemar_p_v2).abs().max() < 1e-12
          and (j.delta_eicr_paired - j.d_eicr_paired).abs().max() < 1e-9)
    print(f"VERIFY 1: unpaired EICR, n_pairs and McNemar equal v2/eicr_reconciliation.csv in 16 cells -> "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT / 'eicr_reconciled.csv'}")


def pull2():
    ev = V.load_e1_all()
    p = V.pairs(ev, T.FIX, T.NEU)
    rows = []
    for m, _ in T.MODELS:
        for d, _ in T.DATASETS:
            x = p[(p.model == m) & (p.dataset == d)]
            dq = (x.t_r - x.t_c).to_numpy()
            both = x[~x.nc_r & ~x.nc_c]
            dqb = (both.t_r - both.t_c).to_numpy()
            n = len(x)
            rows.append({"model": m, "dataset": d, "n_eligible_pairs": n, "n_both_completed": len(both),
                         "n_earlier": int((dq > 0).sum()), "n_tie": int((dq == 0).sum()), "n_later": int((dq < 0).sum()),
                         "asr": 100 * (dq > 0).mean(), "reverse_rate": 100 * (dq < 0).mean(),
                         "median_dq_allpairs": float(np.median(dq)), "median_dq_bothcompleted": float(np.median(dqb)),
                         "mean_dq_bothcompleted": dqb.mean(), "q25_dq_bothcompleted": float(np.percentile(dqb, 25)),
                         "q75_dq_bothcompleted": float(np.percentile(dqb, 75)),
                         "n_noncompletion_cue": int(x.nc_c.sum()), "n_noncompletion_neutral": int(x.nc_r.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "timing_reconciled.csv", index=False)
    print(out.to_string(index=False, float_format=lambda f: f"{f:.3g}"))
    old = pd.read_csv(OUT / "asr_table_all.csv")
    old = old[old.contrast == "Fixed Sufficiency vs Neutral"].merge(out, on=["model", "dataset"])
    ok = (old.n_pairs == old.n_eligible_pairs).all() and (old.pct_earlier - old.asr).abs().max() < 1e-9 and \
        (old.pct_later - old.reverse_rate).abs().max() < 1e-9
    print(f"VERIFY 2: n_eligible_pairs, asr and reverse_rate equal v2/asr_table_all.csv (Fixed vs Neutral) in"
          f" {len(old)}/16 cells -> {'PASS' if ok and len(old) == 16 else 'FAIL'}")
    print(f"wrote {OUT / 'timing_reconciled.csv'}")


if __name__ == "__main__":
    pull1()
    pull2()

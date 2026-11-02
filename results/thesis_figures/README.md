# Final thesis figures

These are the six figures in the final thesis PDF, and only those. Files are
named by thesis figure number. The plots in the result bundles' `figures/`
folders are supporting material, not figures printed in the thesis. Earlier
iterations were removed on 2026-11-02 and remain in git history.

**Verify:** `sha256sum results/thesis_figures/fig*` and compare with the
"Thesis PDF SHA-256" column. ✅ means the repo file is byte-identical to the
file embedded in the thesis.

| File | Thesis | Shows | Thesis PDF SHA-256 | Repo file | Generator |
|---|---|---|---|---|---|
| `fig3_1_agentic_rag_loop.png` | Figure 3.1 | System architecture: the agentic RAG loop and the attack point | `7da9bea5…4417c798` | ⚠️ content match, checksum differs (`414044a1…`) | diagram (no script); same image as `docs/figures/agentic_rag_loop.png` |
| `fig4_1_routes.pdf` | Figure 4.1 | How Neutral episodes end, per model (explicit completion, no-tool exit, technical failure) | `250a015e…f328f98cc1` | ⚠️ content match, checksum differs (`32ca8f37…`) | `python -m prematurestop.analysis.fixed_cue_matrix.thesis_figures` (`fig5_2`) |
| `fig5_1_pcr_curves.pdf` | Figure 5.1 | PCR curves on HotpotQA, one panel per model: Neutral, Fixed Sufficiency, Suff. + Saturation, Continue Search; log x-axis | `8acca759…7a29e` | ⚠️ content match, checksum differs (`4cfee59b…`) | `scripts/thesis_figures/fig5_1_pcr_curves.py` |
| `fig5_2_step_boxplot.png` | Figure 5.2 | Paired step reduction D_q, Fixed Sufficiency vs Neutral, 16 model × dataset combinations | `5a755732…2d977a7` | ✅ identical | `scripts/thesis_figures/fig5_2_step_boxplot.py` |
| `fig5_3_e2_pcr_cdf.pdf` | Figure 5.3 | Bridge-anchored study PCR curves (Neutral, Fixed Sufficiency, Bridge K3) on common anchor-matched questions | `d5f675aa…74fefa91` | ✅ identical | `scripts/thesis_figures/thesis_restructure_e2.py --cov` |
| `fig5_4_e2_forest.png` | Figure 5.4 | Bridge-anchored study forest plot: five head-to-head contrasts with 95% bootstrap CIs, Qwen and Llama, HotpotQA and 2Wiki | `5d086a8f…8462af2a` | ✅ identical | `scripts/thesis_figures/thesis_restructure_e2.py` |

Full thesis checksums:

```text
7da9bea58e3404358ce27e05f3767162958764a3151e2ca287d274044417c798  Figure 3.1  agentic_rag_loop.png
250a015ea6256e9cc28549a31fd8f1fac6f4fc021b8fbd2d9390f5f328f98cc1  Figure 4.1  fig5_2_routes.pdf
8acca759e41ad2572492036d8cfdba4c2153e80fe23b8489ed3a7093d227a29e  Figure 5.1  fig5_1_pcr_curves.pdf
5a7557323cb6eee1dcd970a99e38734c2857ab1b58df9dd67ae9a00ba2d977a7  Figure 5.2  fig5_2_step_boxplot.png
d5f675aa0548f1de85c27e13d9ad8b510ecde1e78238bb5fbf151e3e74fefa91  Figure 5.3  fig_e2_pcr_cdf.pdf
5d086a8fbb51d21f9a72c57b29f1ed85d2f1750f302e13835cf5e9408462af2a  Figure 5.4  fig_e2_forest.png
```

## Reproducibility

* **5.2 and 5.4** regenerate byte-identical from the committed data.
* **5.3** regenerates pixel-identical: the same bytes apart from the PDF's
  embedded `/CreationDate`.
* **5.1** regenerates identical to the repo copy, apart from `/CreationDate`.
* **3.1, 4.1 and 5.1:** the copies embedded in the thesis are not
  byte-identical to the repo files. The difference is not just the
  timestamp. Each is the only candidate matching the figure's content, but
  until the thesis copies are committed here, treat these three as content
  matches only.

Scripts write to `results/figures/` (gitignored scratch), so regenerating them
never overwrites this folder. Data behind the plots is in `data/`.

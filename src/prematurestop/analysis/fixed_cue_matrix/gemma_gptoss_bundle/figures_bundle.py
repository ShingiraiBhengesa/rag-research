"""PCR attack-panel figures for gemma27b and gpt_oss_20b.

Reuses `prematurestop.analysis.fixed_cue_matrix.results_bundle.figures_bundle.plot_pcr_attack_panels`
and `PCR_GRID_ROWS` completely unchanged -- that function is generic over
`model` and (after a small backward-compatible extension to accept an
explicit `datasets` tuple) over dataset coverage too, so gpt_oss_20b's
missing 2wiki column renders as 3 columns, not a blank 4th one.
"""

from __future__ import annotations

from pathlib import Path

from prematurestop.analysis.fixed_cue_matrix.results_bundle.figures_bundle import (
    PCR_GRID_ROWS,
    _read_csv,
    _slugify,
    plot_pcr_attack_panels,
)

from .scope import EXTENDED_BUNDLE_MODELS, MODEL_DATASETS


def generate_pcr_attack_panels(data_dir: Path, fig_root: Path) -> None:
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    for model in EXTENDED_BUNDLE_MODELS:
        for group_label, baseline_cond, attack_conds in PCR_GRID_ROWS:
            plot_pcr_attack_panels(
                stepwise, model, group_label, baseline_cond, attack_conds,
                fig_root / "pcr_curves", f"pcr_attack_{model}_{_slugify(group_label)}",
                datasets=MODEL_DATASETS[model],
            )

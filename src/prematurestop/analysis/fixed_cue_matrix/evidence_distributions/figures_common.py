"""Shared figure constants/helpers -- consistent size, font, category order,
condition naming, and color mapping across every figure in this package."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .source_loader import MODEL_DISPLAY, DATASET_DISPLAY, CONDITION_DISPLAY, DATASETS
from prematurestop.paths import RESULTS_E1

FIGURES_DIR = RESULTS_E1 / "evidence_distributions" / "figures"
SLIDE_DIR = FIGURES_DIR / "slide"
APPENDIX_DIR = FIGURES_DIR / "appendix"

BASE_FONTSIZE = 12
plt.rcParams.update({"font.size": BASE_FONTSIZE})

# Condition order for the primary natural-retrieval figure (Phase 3) --
# Oracle conditions deliberately excluded here, per spec.
NATURAL_CONDITION_ORDER = (
    "NaturalClean",
    "Natural_epistemic_confirmation",
    "Natural_search_match_cue",
    "Natural_source_coverage",
    "Natural_epistemic_confirmation_source",
    "Natural_search_match_source",
    "Natural_PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation",
)
ORACLE_CONDITION_ORDER = ("OracleComplete_Neutral", "OracleComplete_PreRegisteredSufficiencySaturation")

PRIMARY_ECDF_CONDITIONS = (
    "NaturalClean", "Natural_epistemic_confirmation", "Natural_search_match_cue",
    "Natural_PreRegisteredSufficiency", "Natural_PreRegisteredSufficiencySaturation",
)
SOURCE_ABLATION_ECDF_CONDITIONS = (
    "NaturalClean", "Natural_source_coverage", "Natural_epistemic_confirmation",
    "Natural_epistemic_confirmation_source", "Natural_search_match_cue", "Natural_search_match_source",
)

# One fixed color per condition, reused across every figure/model/dataset.
CONDITION_LINE_COLOR = {
    "NaturalClean": "#4D4D4D",
    "Natural_epistemic_confirmation": "#1F77B4",
    "Natural_search_match_cue": "#2CA02C",
    "Natural_source_coverage": "#9B2226",
    "Natural_epistemic_confirmation_source": "#0B4F8A",
    "Natural_search_match_source": "#1B6B1B",
    "Natural_PreRegisteredSufficiency": "#CA6702",
    "Natural_PreRegisteredSufficiencySaturation": "#AE2012",
    "OracleComplete_Neutral": "#6D28D9",
    "OracleComplete_PreRegisteredSufficiencySaturation": "#111827",
}


def save_all_formats(fig, out_path_no_ext: Path, dpi: int = 300) -> None:
    out_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path_no_ext) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_path_no_ext) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path_no_ext) + ".svg", bbox_inches="tight")

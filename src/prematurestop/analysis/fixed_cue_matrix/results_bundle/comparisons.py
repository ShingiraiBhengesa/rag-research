"""Unified condition-comparison manifest.

Single source of truth for every control/cue comparison pair used across
this bundle (`paired_deltas.py`, `retrieval_equivalence_ext.py`,
`answer_ext.py`, `evidence_ext.py`), replacing the two previously
inconsistent, differently-directioned lists
(`canonical_completion.PAIR_SPECS` and `statistics.COMPARISON_FAMILY`) with
one file. Adds the two ablation-vs-single-cue-parent comparisons
(spec section 6's last two rows) that neither prior list encoded --
`condition_metadata.csv`'s own `comparison_baseline` column points those two
ablation conditions at Clean/Neutral, not at their single-cue parent.

Every comparison here is stored as (control_condition, cue_condition) --
i.e. `cue_condition` is always the condition whose effect is being measured,
`control_condition` is what it's compared against. This fixes the direction
ambiguity between the two prior lists (`PAIR_SPECS` stored
(condition, baseline); `COMPARISON_FAMILY` stored (baseline, condition) --
opposite of each other).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_name: str
    control_condition: str
    cue_condition: str
    comparison_role: str
    is_preregistered_primary: bool


# Verbatim from the locked spec section 6 (comparisons table), plus the
# generic "each main natural cue vs NaturalClean" row expanded into one
# comparison per cue condition.
_MAIN_NATURAL_CUES: tuple[str, ...] = (
    "Natural_epistemic_confirmation",
    "Natural_search_match_cue",
    "Natural_PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation",
)

COMPARISONS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "Natural Neutral vs Natural Clean", "NaturalClean", "NaturalNeutral",
        "Added-text control", True,
    ),
    ComparisonSpec(
        "Epistemic Confirmation vs Natural Neutral", "NaturalNeutral", "Natural_epistemic_confirmation",
        "Semantic confirmation effect", True,
    ),
    ComparisonSpec(
        "Search Match vs Natural Neutral", "NaturalNeutral", "Natural_search_match_cue",
        "Search-goal match effect", True,
    ),
    ComparisonSpec(
        "Fixed Sufficiency vs Natural Neutral", "NaturalNeutral", "Natural_PreRegisteredSufficiency",
        "Sufficiency effect", True,
    ),
    ComparisonSpec(
        "Fixed Sufficiency Saturation vs Natural Neutral", "NaturalNeutral", "Natural_PreRegisteredSufficiencySaturation",
        "Sufficiency-plus-saturation effect", True,
    ),
    *[
        ComparisonSpec(
            f"{cue} vs Natural Clean (total end-to-end change)", "NaturalClean", cue,
            "Total end-to-end change", True,
        )
        for cue in _MAIN_NATURAL_CUES
    ],
    ComparisonSpec(
        "Continue Search vs Natural Neutral", "NaturalNeutral", "NaturalContinueSearch",
        "Opposite-direction control", True,
    ),
    ComparisonSpec(
        "Oracle Complete-Saturation vs Oracle Complete-Neutral", "OracleComplete_Neutral",
        "OracleComplete_PreRegisteredSufficiencySaturation",
        "Evidence-complete calibration", True,
    ),
    ComparisonSpec(
        "Source Coverage vs Natural Neutral", "NaturalNeutral", "Natural_source_coverage",
        "Source wording alone", True,
    ),
    # The two currently-missing ablation-vs-single-cue-parent comparisons.
    ComparisonSpec(
        "Epistemic Confirmation + Source Coverage vs Epistemic Confirmation",
        "Natural_epistemic_confirmation", "Natural_epistemic_confirmation_source",
        "Added source-coverage component", True,
    ),
    ComparisonSpec(
        "Search Match + Source Coverage vs Search Match",
        "Natural_search_match_cue", "Natural_search_match_source",
        "Added source-coverage component", True,
    ),
)


def write_condition_comparison_manifest_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["comparison_name", "control_condition", "cue_condition", "comparison_role", "is_preregistered_primary"]
        )
        for c in COMPARISONS:
            writer.writerow([c.comparison_name, c.control_condition, c.cue_condition, c.comparison_role, c.is_preregistered_primary])

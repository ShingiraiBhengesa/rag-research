"""Step 12 (workbook): `qwen_llama_results_workbook.xlsx`.

One sheet per `data/*.csv`, via `pandas.ExcelWriter(engine="openpyxl")`.
Requires `openpyxl` (added to `environments/analysis.txt`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Excel worksheet names are capped at 31 characters and can't contain
# some punctuation -- map the longer CSV stems to short, valid names.
_SHEET_NAME_OVERRIDES = {
    "condition_comparison_manifest": "comparison_manifest",
    "pcr_paired_deltas_stepwise": "pcr_paired_deltas",
    "route_decomposition_stepwise": "route_decomposition",
    "evidence_run_level": "evidence_run_level",
    "answer_paired_flips": "answer_paired_flips",
}


def _sheet_name(stem: str) -> str:
    name = _SHEET_NAME_OVERRIDES.get(stem, stem)
    return name[:31]


def write_workbook(data_dir: Path, path: Path) -> None:
    csv_paths = sorted(data_dir.glob("*.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=_sheet_name(csv_path.stem), index=False)

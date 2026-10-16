"""Split file locations (dependency-free so every phase can import it)."""

from __future__ import annotations

from pathlib import Path

from prematurestop.paths import EXPERIMENT_E2, REPO_ROOT  # noqa: F401
DEFAULT_SPLITS = EXPERIMENT_E2 / "splits"


def split_paths(split_dir: Path, dataset: str, split: str) -> dict[str, Path]:
    base = Path(split_dir) / dataset / split
    return {k: base / f"{dataset}_{split}_{k}.jsonl" for k in ("manifest", "corpus", "evidence")}

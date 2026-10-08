"""Split file locations (dependency-free so every phase can import it)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "experiments" / "dynamic_bridge_cue_pilot" / "splits"


def split_paths(split_dir: Path, dataset: str, split: str) -> dict[str, Path]:
    base = Path(split_dir) / dataset / split
    return {k: base / f"{dataset}_{split}_{k}.jsonl" for k in ("manifest", "corpus", "evidence")}

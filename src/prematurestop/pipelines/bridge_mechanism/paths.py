"""Split file locations (dependency-free so every phase can import it)."""

from __future__ import annotations

from pathlib import Path

from prematurestop.paths import EXPERIMENT_E2, REPO_ROOT  # noqa: F401
DEFAULT_SPLITS = EXPERIMENT_E2 / "splits"


def split_paths(split_dir: Path, dataset: str, split: str) -> dict[str, Path]:
    base = Path(split_dir) / dataset / split
    return {k: base / f"{dataset}_{split}_{k}.jsonl" for k in ("manifest", "corpus", "evidence")}


# Records written before the 2026-10-16 restructure store absolute trace paths
# under the old state directory. The state now lives in the local data root.
_OLD_STATE_MARKER = "/experiments/dynamic_bridge_cue_pilot/state/"


def resolve_state_path(stored: str | Path) -> Path:
    """Locate a file whose path was stored in an E2 state record.

    Paths written before the restructure point at
    `<old repo>/experiments/dynamic_bridge_cue_pilot/state/...`; they map to
    `$PREMATURESTOP_LOCAL_DATA/e2_bridge_mechanism/state/...`. Any other path is
    returned unchanged.
    """
    from prematurestop.paths import LOCAL_DATA

    s = str(stored)
    if _OLD_STATE_MARKER in s and not Path(s).exists():
        return LOCAL_DATA / "e2_bridge_mechanism" / "state" / s.split(_OLD_STATE_MARKER, 1)[1]
    return Path(s)

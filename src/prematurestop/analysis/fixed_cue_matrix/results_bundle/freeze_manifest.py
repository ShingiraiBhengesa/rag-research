"""Step 1: `analysis_manifest.json`.

Reuses `completion_inventory.build_inventory`'s already-fetched
`RowRecord.frozen_config_hashes` (populated per-row from the production
row's own `frozen_config_hashes` field, written at production time) rather
than re-hashing input files from scratch, and its `CellInventory`/
`fetch_active_jobs_readonly` for the shard-completion and active-job
confirmation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from prematurestop.evaluation import config
from prematurestop.evaluation.completion_inventory import CellInventory, InventoryResult, RowRecord

from .scope import CANONICAL_BUNDLE_MODELS


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _unique_hashes(rows: list[RowRecord]) -> dict[str, dict[str, list[str]]]:
    """(model, dataset) -> {hash_field: [unique values seen]}. Multiple
    distinct values for one (model, dataset) would indicate the input files
    changed mid-run -- reported, not silently averaged over."""
    out: dict[str, dict[str, set[str]]] = {}
    for r in rows:
        key = f"{r.model}/{r.dataset}"
        bucket = out.setdefault(key, {})
        for field, value in (r.frozen_config_hashes or {}).items():
            bucket.setdefault(field, set()).add(str(value))
    return {k: {f: sorted(v) for f, v in fields.items()} for k, fields in out.items()}


def build_manifest(inventory: InventoryResult, repo_root: Path, generated_at: str) -> dict:
    active_for_included = [
        j for j in inventory.active_jobs
        if any(m in j.get("job_name", "") for m in CANONICAL_BUNDLE_MODELS) and "error" not in j
    ]
    return {
        "generated_at": generated_at,
        "git_commit": _git_head(repo_root),
        "included_models": list(CANONICAL_BUNDLE_MODELS),
        "datasets": list(config.DATASETS),
        "conditions": list(config.CONDITIONS),
        "production_root": str(config.DEFAULT_PRODUCTION_ROOT),
        "input_file_hashes_by_model_dataset": _unique_hashes(inventory.rows),
        "shard_completion": [
            {
                "model": c.model, "dataset": c.dataset, "condition": c.condition,
                "n_shards_expected": c.n_shards_expected, "n_shards_present": c.n_shards_present,
                "status": c.status,
            }
            for c in inventory.cells
        ],
        "active_slurm_jobs_for_included_models": active_for_included,
        "note": (
            "active_slurm_jobs_for_included_models must be empty for qwen14b/llama8b "
            "-- both models are frozen/complete inputs for this bundle."
        ),
    }


def write_analysis_manifest_json(inventory: InventoryResult, repo_root: Path, generated_at: str, path: Path) -> dict:
    manifest = build_manifest(inventory, repo_root, generated_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

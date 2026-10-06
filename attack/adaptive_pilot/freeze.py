"""Freeze and hash the pilot procedure before held-out evaluation.

Produces one manifest with a SHA-256 over every component that must not change
between development and held-out runs: cue specs, prompts, attacker config,
length limit, feedback schema/access, retry policy, two-attempt budget, dataset
split manifests, and the pilot code modules themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attack.adaptive_pilot import config as pilot_config
from attack.adaptive_pilot.attacker import prompt_version_hashes
from attack.adaptive_pilot.cue_specs import DEFAULT_SUFFIX_TOKEN_LIMIT, specs_document
from attack.adaptive_pilot.feedback import (
    ACCESS_SETTINGS,
    FEEDBACK_FIELDS_V1,
    FEEDBACK_SCHEMA_VERSION,
    describe_information_content,
)
from attack.runtime.hashing import hash_text, stable_hash

PKG_DIR = Path(__file__).resolve().parent
CODE_MODULES = (
    "attacker.py", "cue_specs.py", "feedback.py", "runner.py",
    "split_manifest.py", "validation.py", "config.py", "freeze.py",
)
# Canonical files the pilot depends on read-only; hashed so a canonical change
# that would affect the pilot is detectable.
CANONICAL_DEPENDENCIES = (
    PKG_DIR.parent / "runtime" / "cues.py",
    PKG_DIR.parent / "cue_definitions.py",
    PKG_DIR.parent / "runtime" / "benchmark_search.py",
    PKG_DIR.parent / "runtime" / "runner.py",
    PKG_DIR.parents[1] / "analysis" / "canonical" / "canonical_completion.py",
)


def _file_hash(path: Path) -> str:
    return hash_text(path.read_text(encoding="utf-8"), length=None)


def build_freeze_manifest(*, split_dir: Path, suffix_token_limit: int = DEFAULT_SUFFIX_TOKEN_LIMIT) -> dict:
    split_manifests = sorted(split_dir.glob("**/*_manifest.jsonl"))
    manifest = {
        "schema_version": "adaptive_cue_pilot_freeze_v1",
        "cue_specs": specs_document(suffix_token_limit=suffix_token_limit),
        "suffix_token_limit": suffix_token_limit,
        "prompt_version_hashes": prompt_version_hashes(),
        "attacker": {
            "model_key": pilot_config.ATTACKER_MODEL_KEY,
            "model_id": pilot_config.VICTIM_MODELS[pilot_config.ATTACKER_MODEL_KEY]["model_id"],
            "revision": pilot_config.VICTIM_MODELS[pilot_config.ATTACKER_MODEL_KEY]["revision"],
            "temperature": pilot_config.ATTACKER_TEMPERATURE,
            "top_p": pilot_config.ATTACKER_TOP_P,
        },
        "feedback": {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "fields": list(FEEDBACK_FIELDS_V1),
            "access_settings": list(ACCESS_SETTINGS),
            "information_content": describe_information_content(),
        },
        "retry_policy": {"max_technical_retries": pilot_config.MAX_TECHNICAL_RETRIES, "retry_on": list(pilot_config.RETRY_ON)},
        "two_attempt_budget": {"attempts": 2, "selection_rule": "revised_payload_is_final", "shared_control": True},
        "launch_conditions": list(pilot_config.LAUNCH_CONDITION_IDS),
        "victims": list(pilot_config.PILOT_VICTIMS),
        "victim_settings": {
            "temperature": pilot_config.VICTIM_TEMPERATURE,
            "seed": pilot_config.VICTIM_SEED,
            "top_k": pilot_config.TOP_K,
            "emergency_max_react_tool_calls": pilot_config.EMERGENCY_MAX_REACT_TOOL_CALLS,
            "max_researcher_iterations": pilot_config.MAX_RESEARCHER_ITERATIONS,
        },
        "split_manifest_hashes": {str(p.relative_to(split_dir)): _file_hash(p) for p in split_manifests},
        "code_module_hashes": {m: _file_hash(PKG_DIR / m) for m in CODE_MODULES},
        "canonical_dependency_hashes": {p.name: _file_hash(p) for p in CANONICAL_DEPENDENCIES if p.exists()},
    }
    manifest["freeze_sha256"] = stable_hash(manifest, length=None)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the adaptive-cue pilot before held-out evaluation.")
    ap.add_argument("--split-dir", type=Path, default=pilot_config_default_split_dir())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix-token-limit", type=int, default=DEFAULT_SUFFIX_TOKEN_LIMIT)
    args = ap.parse_args()
    manifest = build_freeze_manifest(split_dir=args.split_dir, suffix_token_limit=args.suffix_token_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"freeze_sha256={manifest['freeze_sha256']}")
    print(f"written: {args.out}")
    return 0


def pilot_config_default_split_dir() -> Path:
    from attack.adaptive_pilot.split_manifest import DEFAULT_OUT

    return DEFAULT_OUT


if __name__ == "__main__":
    raise SystemExit(main())

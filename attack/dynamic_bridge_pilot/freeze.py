"""Hash manifest of the whole scientific procedure, frozen before held-out.

Covers: the config snapshot, every source file and prompt of this package, the
canonical runtime files the victim path depends on, the frozen lambda, and
the held-out split manifests. `verify()` recomputes the manifest and returns
every mismatch; the held-out arms phase refuses to start unless it is empty,
so NRP runs exactly the frozen procedure.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from attack.dynamic_bridge_pilot import config as C
from attack.runtime.hashing import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
CANONICAL_DEPENDENCIES = (
    "attack/runtime/cues.py", "attack/runtime/runner.py", "attack/runtime/benchmark_search.py",
    "attack/adaptive_pilot/runner.py", "attack/adaptive_pilot/validation.py",
    "analysis/canonical/canonical_completion.py", "analysis/evidence.py", "analysis/trace_parser.py",
    "attack/answer_scoring/answer_evaluator_v2.py",
    "open_deep_research/src/open_deep_research/deep_researcher.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_files() -> list[Path]:
    files = [p for p in PACKAGE_DIR.rglob("*") if p.is_file() and p.suffix in (".py", ".txt")
             and "tests" not in p.parts and "__pycache__" not in p.parts]
    return sorted(files)


def build_manifest(*, lambda_path: Path, split_manifests: list[Path]) -> dict:
    files = {str(p.relative_to(REPO_ROOT)): _sha(p) for p in _package_files()}
    deps = {rel: _sha(REPO_ROOT / rel) for rel in CANONICAL_DEPENDENCIES}
    lam = json.loads(lambda_path.read_text(encoding="utf-8"))
    splits = {str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p): _sha(p)
              for p in sorted(split_manifests)}
    body = {
        "pilot_version": C.PILOT_VERSION,
        "config_snapshot_hash": stable_hash(C.frozen_snapshot(), length=None),
        "config_snapshot": C.frozen_snapshot(),
        "package_files": files,
        "canonical_dependencies": deps,
        "lambda": lam["lambda"],
        "lambda_hash": lam["lambda_hash"],
        "lambda_source": lam.get("source"),
        "split_manifests": splits,
    }
    body["manifest_hash"] = stable_hash({k: v for k, v in body.items() if k != "config_snapshot"}, length=None)
    try:
        body["git_head_at_freeze"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                                    text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        body["git_head_at_freeze"] = ""
    return body


def write(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def verify(frozen_path: Path, *, lambda_path: Path, split_manifests: list[Path]) -> list[str]:
    if not frozen_path.exists():
        return [f"no freeze manifest at {frozen_path}"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    now = build_manifest(lambda_path=lambda_path, split_manifests=split_manifests)
    problems = []
    for section in ("package_files", "canonical_dependencies", "split_manifests"):
        a, b = frozen.get(section, {}), now.get(section, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                problems.append(f"{section}:{k} changed")
    for k in ("config_snapshot_hash", "lambda_hash"):
        if frozen.get(k) != now.get(k):
            problems.append(f"{k} changed")
    return problems

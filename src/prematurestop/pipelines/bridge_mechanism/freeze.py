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

from prematurestop.pipelines.bridge_mechanism import config as C
from prematurestop.infra.hashing import stable_hash

from prematurestop.paths import REPO_ROOT  # noqa: E402
PACKAGE_DIR = Path(__file__).resolve().parent
# Code outside this pipeline that the victim path and the outcome definitions
# depend on. (Freeze manifests written before the 2026-10 repository
# reorganisation list the same files under their old paths; the frozen run
# commits are the provenance record for those runs.)
CANONICAL_DEPENDENCIES = (
    "src/prematurestop/cues/fixed/registry.py", "src/prematurestop/system/runner.py",
    "src/prematurestop/system/benchmark_search.py", "src/prematurestop/system/run_helpers.py",
    "src/prematurestop/cues/dynamic/text_checks.py", "src/prematurestop/evaluation/completion.py",
    "src/prematurestop/evaluation/evidence.py", "src/prematurestop/evaluation/trace_parser.py",
    "src/prematurestop/evaluation/answer_scoring.py",
    "third_party/open_deep_research/src/open_deep_research/deep_researcher.py",
    "third_party/open_deep_research/src/open_deep_research/utils.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_files() -> list[Path]:
    dirs = (PACKAGE_DIR, REPO_ROOT / "src" / "prematurestop" / "cues" / "dynamic")
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and p.suffix in (".py", ".txt")
             and "__pycache__" not in p.parts]
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

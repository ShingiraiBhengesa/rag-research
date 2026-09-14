"""Second, top-level reorganization: drop the "v3"/"whitebox_" versioning
entirely and produce a clean, publication-facing repo layout:

    attack/     (was whitebox_attack/v3/{runtime,launch,legacy_evaluators,dev_tools,tests}
                 + whitebox_attack/{metrics.py,cue_definitions.py,__init__.py})
    analysis/   (was whitebox_attack/v3/analysis/*; canonical_completion_v1/ -> canonical/)
    results/    (was whitebox_paper_results/e2e_full_system_v3/production/*)
    archive/    (was whitebox_attack/archive/* + old whitebox_paper_results v1/v2/pcr dirs)

SAFETY: defaults to --dry-run. Pass --execute to actually move files and
rewrite imports/path-string-literals. Never touches `.git/`, `open_deep_research/`
(only its one cross-import line), or anything already under `archive/`.

Run from the repo root:
    python3 whitebox_attack/v3/analysis/canonical_completion_v1/reorg_prep/reorg_top_level.py --dry-run
    python3 whitebox_attack/v3/analysis/canonical_completion_v1/reorg_prep/reorg_top_level.py --execute

After --execute you MUST:
1. Fix the two DEFAULT_PRODUCTION_ROOT definitions (analysis/config.py,
   attack/runtime/production_manifest.py) and the prior_sample_overlap()
   legacy-path dict (attack/runtime/production_manifest.py) by hand --
   these construct paths via Path(...) / "segment" / "segment", not a
   single string literal, so this script does not touch them.
2. Run the full test suites.
3. Regenerate sbatch scripts (`python -m attack.launch.generate_production_arrays --mode generate`)
   and validate (`--mode validate`).
4. Update README.md / PUBLISHING.md / .gitignore by hand (prose, not mechanical).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# --- Python import rewrites (old dotted module -> new dotted module) -------
# Order matters: more specific (longer) prefixes must be checked before
# shorter ones, so canonical_completion_v1 rewrites before the general
# whitebox_attack.v3.analysis rewrite.
IMPORT_PREFIX_MOVES: list[tuple[str, str]] = [
    ("whitebox_attack.v3.analysis.canonical_completion_v1", "analysis.canonical"),
    ("whitebox_attack.v3.analysis", "analysis"),
    ("whitebox_attack.v3.runtime", "attack.runtime"),
    ("whitebox_attack.v3.launch", "attack.launch"),
    ("whitebox_attack.v3.legacy_evaluators", "attack.legacy_evaluators"),
    ("whitebox_attack.v3.dev_tools", "attack.dev_tools"),
    ("whitebox_attack.v3.tests", "attack.tests"),
    ("whitebox_attack.metrics", "attack.metrics"),
    ("whitebox_attack.cue_definitions", "attack.cue_definitions"),
]

# --- Literal repo-relative path strings to rewrite (fingerprint hash lists,
# docs, etc.) -- longest-prefix-first so e.g. the analysis/canonical path
# doesn't get double-rewritten by the shorter attack/... rule.
LITERAL_PATH_MOVES: list[tuple[str, str]] = [
    ("whitebox_attack/v3/analysis/canonical_completion_v1", "analysis/canonical"),
    ("whitebox_attack/v3/analysis", "analysis"),
    ("whitebox_attack/v3/runtime", "attack/runtime"),
    ("whitebox_attack/v3/launch", "attack/launch"),
    ("whitebox_attack/v3/legacy_evaluators", "attack/legacy_evaluators"),
    ("whitebox_attack/v3/dev_tools", "attack/dev_tools"),
    ("whitebox_attack/v3/tests", "attack/tests"),
    ("whitebox_attack/metrics.py", "attack/metrics.py"),
    ("whitebox_attack/cue_definitions.py", "attack/cue_definitions.py"),
    ("whitebox_attack/v3", "attack"),  # any remaining bare v3 doc reference
]

# --- Directory/file moves (source relative to REPO_ROOT -> dest relative to REPO_ROOT) ---
# ORDER MATTERS: a parent directory must move to its final location BEFORE
# any of its own subdirectories are moved again -- shutil.move() moves a
# source INSIDE an existing destination directory rather than merging, so
# moving analysis/canonical_completion_v1 before its parent `analysis` had
# already been relocated would create a broken nested analysis/analysis/.
DIRECTORY_MOVES: list[tuple[str, str]] = [
    ("whitebox_attack/v3/analysis", "analysis"),
    ("analysis/canonical_completion_v1", "analysis/canonical"),  # source now post-first-move
    ("whitebox_attack/v3/runtime", "attack/runtime"),
    ("whitebox_attack/v3/launch", "attack/launch"),
    ("whitebox_attack/v3/legacy_evaluators", "attack/legacy_evaluators"),
    ("whitebox_attack/v3/dev_tools", "attack/dev_tools"),
    ("whitebox_attack/v3/tests", "attack/tests"),
    ("whitebox_attack/archive", "archive/code"),
    ("whitebox_paper_results/e2e_final_answer_v1", "archive/legacy_results/e2e_final_answer_v1"),
    ("whitebox_paper_results/e2e_full_system_v1", "archive/legacy_results/e2e_full_system_v1"),
    ("whitebox_paper_results/e2e_full_system_v2", "archive/legacy_results/e2e_full_system_v2"),
    ("whitebox_paper_results/pcr_full_completion_v1", "archive/legacy_results/pcr_full_completion_v1"),
    ("whitebox_paper_results/pcr_full_completion_v2_60", "archive/legacy_results/pcr_full_completion_v2_60"),
    ("whitebox_paper_results/e2e_full_system_v3/production", "results"),
]

FILE_MOVES: list[tuple[str, str]] = [
    ("whitebox_attack/v3/__init__.py", "attack/__init__.py"),
    ("whitebox_attack/metrics.py", "attack/metrics.py"),
    ("whitebox_attack/cue_definitions.py", "attack/cue_definitions.py"),
]

EXCLUDE_DIR_MARKERS = ("/.git/", "/archive/")


def _should_scan(path: Path) -> bool:
    s = str(path)
    if any(marker in s for marker in EXCLUDE_DIR_MARKERS):
        return False
    return True


def _rewrite_text(text: str) -> tuple[str, list[str]]:
    changes = []
    new_text = text
    for old, new in IMPORT_PREFIX_MOVES:
        # `from OLD.rest import ...` and `from OLD import ...` and bare `OLD.rest`
        pattern_from = re.compile(rf"\bfrom {re.escape(old)}\b")
        if pattern_from.search(new_text):
            new_text = pattern_from.sub(f"from {new}", new_text)
            changes.append(f"from {old}... -> from {new}...")
        pattern_bare = re.compile(rf"(?<![\w.]){re.escape(old)}\.")
        if pattern_bare.search(new_text):
            new_text = pattern_bare.sub(f"{new}.", new_text)
            changes.append(f"{old}.X -> {new}.X (bare dotted reference)")
    for old, new in LITERAL_PATH_MOVES:
        if old in new_text:
            new_text = new_text.replace(old, new)
            changes.append(f'path string "{old}" -> "{new}"')
    return new_text, changes


# Only these top-level trees can possibly contain a reference needing
# rewriting (code + docs). Deliberately excludes `whitebox_paper_results/`
# (43GB of trace/result data, no .py/.md/.sh files worth rewriting there --
# it is moved as a directory below, not content-scanned) and
# `chapter3_methodology_audit/` (a frozen historical snapshot, left as-is).
SCAN_ROOTS = ("whitebox_attack", "open_deep_research", ".")  # "." = repo-root-only files, handled separately


def _iter_scan_files():
    for sub in ("whitebox_attack", "open_deep_research"):
        base = REPO_ROOT / sub
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".sh") and _should_scan(p):
                yield p
    # repo-root-level doc files only (not a recursive walk)
    for name in ("README.md", "PUBLISHING.md"):
        p = REPO_ROOT / name
        if p.exists():
            yield p


def plan(execute: bool) -> None:
    all_files = [p for p in _iter_scan_files() if "reorg_top_level.py" not in str(p)]

    file_changes: dict[Path, list[str]] = {}
    for path in all_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text, changes = _rewrite_text(text)
        if changes:
            file_changes[path] = changes
            if execute:
                path.write_text(new_text, encoding="utf-8")

    print(f"=== {'EXECUTING' if execute else 'DRY RUN'}: rewrites in {len(file_changes)} files ===")
    for path, changes in sorted(file_changes.items()):
        print(f"\n{path.relative_to(REPO_ROOT)}:")
        for c in sorted(set(changes)):
            print(f"  - {c}")

    print(f"\n=== {'EXECUTING' if execute else 'DRY RUN'}: directory moves ===")
    for old, new in DIRECTORY_MOVES:
        src = REPO_ROOT / old
        dst = REPO_ROOT / new
        print(f"mv {old} -> {new}")
        if execute:
            if not src.exists():
                print(f"  !! SKIPPED (source not found): {src}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    print(f"\n=== {'EXECUTING' if execute else 'DRY RUN'}: file moves ===")
    for old, new in FILE_MOVES:
        src = REPO_ROOT / old
        dst = REPO_ROOT / new
        print(f"mv {old} -> {new}")
        if execute:
            if not src.exists():
                print(f"  !! SKIPPED (source not found): {src}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if execute:
        for pkg_dir in ("attack", "analysis"):
            init = REPO_ROOT / pkg_dir / "__init__.py"
            if not init.exists():
                init.parent.mkdir(parents=True, exist_ok=True)
                init.write_text("", encoding="utf-8")
                print(f"created missing {pkg_dir}/__init__.py")

    print(
        "\nNEXT STEPS (not done by this script): "
        "fix DEFAULT_PRODUCTION_ROOT in analysis/config.py and "
        "attack/runtime/production_manifest.py, fix prior_sample_overlap()'s "
        "legacy-path dict, run the full test suites, regenerate + validate "
        "sbatch scripts, update README.md/PUBLISHING.md/.gitignore by hand."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan(execute=args.execute)

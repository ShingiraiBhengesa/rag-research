"""Reorganizes attack/'s flat ~25-file layout into subpackages
for publication readiness, and rewrites every cross-reference repo-wide.

SAFETY: defaults to --dry-run (prints every planned change, writes nothing).
Pass --execute to actually move files and rewrite imports. Never touches
`whitebox_attack/archive/`, `.git/`, or any file under
`whitebox_paper_results/` other than the sbatch-generation template strings
inside the (moved) launcher script itself.

Run from the repo root:
    python3 analysis/canonical/reorg_prep/reorg_v3_package.py --dry-run
    python3 analysis/canonical/reorg_prep/reorg_v3_package.py --execute

After --execute, you MUST:
1. Run the full test suites (attack/tests/, analysis/tests/,
   analysis/canonical/test_canonical_completion.py).
2. Regenerate sbatch scripts: python -m attack.launch.generate_production_arrays --mode generate
3. Validate: python -m attack.launch.generate_production_arrays --mode validate
4. Only then consider resubmitting any production job.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
V3 = REPO_ROOT / "whitebox_attack" / "v3"

# old top-level v3/<name>.py -> new subpackage
MODULE_MOVES: dict[str, str] = {
    # runtime/ -- locked, imported live by SLURM jobs
    "production_runner": "runtime/production_runner",
    "runner": "runtime/runner",
    "benchmark_search": "runtime/benchmark_search",
    "runtime_gate": "runtime/runtime_gate",
    "trace_schema": "runtime/trace_schema",
    "hashing": "runtime/hashing",
    "cues": "runtime/cues",
    "corpus": "runtime/corpus",
    "evidence": "runtime/evidence",
    "production_manifest": "runtime/production_manifest",
    "production_models": "runtime/production_models",
    "musique_scope_audit": "runtime/musique_scope_audit",
    # launch/ -- array generation / sbatch templating (renamed for clarity)
    "generate_v3_qwen_cluster_arrays": "launch/generate_production_arrays",
    # legacy_evaluators/ -- pre-canonical, historical, superseded by
    # analysis/ (and analysis/canonical_completion_v1/ for PCR)
    "generate_pcr_cdf_tables": "legacy_evaluators/generate_pcr_cdf_tables",
    "evaluate_completed_shard": "legacy_evaluators/evaluate_completed_shard",
    "final_production_evaluation": "legacy_evaluators/final_production_evaluation",
    "qwen14b_architecture_mechanism_audit": "legacy_evaluators/qwen14b_architecture_mechanism_audit",
    "rescore_completed_answer_v2": "legacy_evaluators/rescore_completed_answer_v2",
    "answer_evaluator_v2": "legacy_evaluators/answer_evaluator_v2",
    "hop_aware_evaluation": "legacy_evaluators/hop_aware_evaluation",
    # dev_tools/ -- one-off dev/calibration scripts, not part of the frozen
    # production pipeline and never imported by any sbatch-launched process
    "cue_development_report": "dev_tools/cue_development_report",
    "dev_validation_data": "dev_tools/dev_validation_data",
    "odr_smoke": "dev_tools/odr_smoke",
    "qwen_gate": "dev_tools/qwen_gate",
    "retriever_calibration": "dev_tools/retriever_calibration",
}

NEW_PACKAGES = ("runtime", "launch", "legacy_evaluators", "dev_tools")

# Literal repo-relative path strings referenced as data (not as python
# imports) that must also be rewritten -- e.g. generate_v3_qwen_cluster_arrays.py's
# frozen_pipeline_fingerprint() source_groups dict, and any f"attack/{x}.py"
# style construction.
LITERAL_PATH_OLD_TO_NEW = {
    f"attack/{old}.py": f"attack/{new}.py" for old, new in MODULE_MOVES.items()
}

# `python -m whitebox_attack.v3.<old>` module-invocation strings inside the
# sbatch template (rendered by the launcher script itself).
MODULE_INVOCATION_OLD_TO_NEW = {
    f"whitebox_attack.v3.{old}": f"whitebox_attack.v3.{new.replace('/', '.')}" for old, new in MODULE_MOVES.items()
}


def _import_regex_replacements(text: str) -> tuple[str, list[str]]:
    changes = []
    new_text = text

    # from whitebox_attack.v3.<old> import ...
    for old, new in MODULE_MOVES.items():
        new_dotted = new.replace("/", ".")
        pattern = re.compile(rf"\bfrom whitebox_attack\.v3\.{re.escape(old)} import\b")
        if pattern.search(new_text):
            new_text = pattern.sub(f"from whitebox_attack.v3.{new_dotted} import", new_text)
            changes.append(f"from whitebox_attack.v3.{old} import -> from whitebox_attack.v3.{new_dotted} import")
        # from whitebox_attack.v3.<old> import (multi-line paren form) already covered by same pattern

    # from whitebox_attack.v3 import <old>[, <old2>, ...] (bare submodule import)
    bare_pattern = re.compile(r"\bfrom whitebox_attack\.v3 import ([a-zA-Z0-9_,\s]+)")

    def _bare_sub(m: re.Match) -> str:
        names = [n.strip() for n in m.group(1).split(",")]
        replaced_lines = []
        for name in names:
            if name in MODULE_MOVES:
                new_dotted_pkg = MODULE_MOVES[name].rsplit("/", 1)[0]
                replaced_lines.append(f"from whitebox_attack.v3.{new_dotted_pkg} import {name}")
                changes.append(f"from whitebox_attack.v3 import {name} -> from whitebox_attack.v3.{new_dotted_pkg} import {name}")
            else:
                replaced_lines.append(f"from whitebox_attack.v3 import {name}")
        return "\n".join(replaced_lines)

    new_text = bare_pattern.sub(_bare_sub, new_text)

    # literal path strings (fingerprint source_groups, etc.)
    for old_path, new_path in LITERAL_PATH_OLD_TO_NEW.items():
        if old_path in new_text:
            new_text = new_text.replace(old_path, new_path)
            changes.append(f'"{old_path}" -> "{new_path}"')

    # `python -m whitebox_attack.v3.<old>` / bare dotted module strings
    for old_mod, new_mod in MODULE_INVOCATION_OLD_TO_NEW.items():
        if old_mod in new_text:
            new_text = new_text.replace(old_mod, new_mod)
            changes.append(f"{old_mod} -> {new_mod}")

    return new_text, changes


def plan(execute: bool) -> None:
    # 1. Determine every .py file in the repo to scan for import rewrites
    #    (excluding archive/, .git, and this reorg script itself).
    all_py_files = [
        p
        for p in REPO_ROOT.rglob("*.py")
        if "/.git/" not in str(p)
        and "/archive/" not in str(p)
        and "reorg_v3_package.py" not in str(p)
    ]

    file_changes: dict[Path, list[str]] = {}
    for path in all_py_files:
        text = path.read_text(encoding="utf-8")
        new_text, changes = _import_regex_replacements(text)
        if changes:
            file_changes[path] = changes
            if execute:
                path.write_text(new_text, encoding="utf-8")

    print(f"=== {'EXECUTING' if execute else 'DRY RUN'}: import/reference rewrites in {len(file_changes)} files ===")
    for path, changes in sorted(file_changes.items()):
        print(f"\n{path.relative_to(REPO_ROOT)}:")
        for c in changes:
            print(f"  - {c}")

    # 2. Move the actual files (after rewriting imports, so the moved file's
    #    own content is already correct).
    print(f"\n=== {'EXECUTING' if execute else 'DRY RUN'}: file moves ===")
    for pkg in NEW_PACKAGES:
        target_dir = V3 / pkg
        print(f"mkdir -p {target_dir.relative_to(REPO_ROOT)}  (+ __init__.py)")
        if execute:
            target_dir.mkdir(parents=True, exist_ok=True)
            init = target_dir / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")

    for old_name, new_rel in MODULE_MOVES.items():
        src = V3 / f"{old_name}.py"
        dst = V3 / f"{new_rel}.py"
        print(f"mv {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")
        if execute:
            if not src.exists():
                print(f"  !! SKIPPED (source not found): {src}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

    print(
        "\nNEXT STEPS (not done by this script): "
        "run the full test suites, "
        "python -m attack.launch.generate_production_arrays --mode generate, "
        "then --mode validate, before resubmitting any production job."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan(execute=args.execute)

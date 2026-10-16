"""Shared helpers for drawing question splits that avoid previously used questions.

Extracted unchanged from the archived adaptive-cue pilot split builder.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from prematurestop.paths import LOCAL_DEV_BUNDLE, LOCAL_MANIFESTS

_PRIOR_SOURCES = {
    "production_n450": lambda ds: [LOCAL_MANIFESTS / f"{ds}_n450.jsonl"],
    "cue_development": lambda ds: [LOCAL_DEV_BUNDLE / "dev_examples.jsonl"],
    "runtime_gate": lambda ds: [LOCAL_DEV_BUNDLE / "smoke_examples.jsonl"],
}


def _ids_from(path: Path, dataset: str) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") in (dataset, None) or str(path).endswith("_n450.jsonl"):
            eid = row.get("example_id")
            if eid:
                ids.add(str(eid))
    return ids


def _prior_ids(dataset: str) -> dict[str, set[str]]:
    return {label: set().union(*[_ids_from(p, dataset) for p in paths(dataset)]) if paths(dataset) else set()
            for label, paths in _PRIOR_SOURCES.items()}


def _question_type(dataset: str, example) -> str:
    return str(example.get("type") or "unspecified")


def _stratified_sample(candidates: list, n: int, seed: int, dataset: str) -> list:
    by_type: dict[str, list] = {}
    for cand in candidates:
        by_type.setdefault(_question_type(dataset, cand.example), []).append(cand)
    for key in by_type:
        by_type[key] = sorted(by_type[key], key=lambda c: c.source_id)
        random.Random(seed + sum(ord(ch) for ch in key)).shuffle(by_type[key])
    picked: list = []
    types = sorted(by_type)
    idx = 0
    # round-robin across types for a type mixture, deterministic
    while len(picked) < n and any(by_type[t] for t in types):
        t = types[idx % len(types)]
        if by_type[t]:
            picked.append(by_type[t].pop())
        idx += 1
    return picked[:n]

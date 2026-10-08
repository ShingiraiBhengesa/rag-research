"""Append-only, fsync'd JSONL checkpoint store (preemption-safe).

Every phase writes one line per completed unit (question or question x arm)
and fsyncs before moving on, so a pod preempted at any point resumes by
skipping keys already on disk. A partially written trailing line (killed
mid-write) is ignored on read and the unit is redone; every unit is
deterministic given its inputs, so redoing it is safe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line from a preempted write
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def done_keys(path: Path, key: str = "key") -> set[str]:
    return {r[key] for r in read_jsonl(path) if key in r}


def index_by(records: Iterable[dict], key: str = "key") -> dict[str, dict]:
    return {r[key]: r for r in records}

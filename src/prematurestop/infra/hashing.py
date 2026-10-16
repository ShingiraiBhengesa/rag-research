"""Stable hashing helpers for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation for hashing and trace comparison."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, *, length: int | None = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def hash_text(text: str, *, length: int | None = 16) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def content_addressed_path(root: Path, kind: str, content_hash: str, suffix: str) -> Path:
    return root / kind / content_hash[:2] / f"{content_hash}{suffix}"


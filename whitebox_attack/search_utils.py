"""
search_utils.py — DuckDuckGo search with atomic file-based cache.

Each result is cached by sha256(query) in CACHE_DIR so:
  - Repeated runs don't re-query DDG (deterministic replay).
  - Multiple concurrent sbatch jobs share the same cache safely
    (atomic rename ensures no partial writes are ever read).

Environment variables
---------------------
DDG_CACHE_DIR : path to cache directory  (default: whitebox_results/ddg_cache)
DDG_SLEEP     : seconds to sleep before each uncached call  (default: 1.5)
DDG_MAX       : max DDG results per query  (default: 3)
"""
import os
import hashlib
import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get(
    "DDG_CACHE_DIR",
    os.path.join(_HERE, "..", "whitebox_results", "ddg_cache"),
)
_SLEEP    = float(os.environ.get("DDG_SLEEP", "1.5"))
_MAX      = int(os.environ.get("DDG_MAX", "3"))
_FALLBACK = "No relevant results found."


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _cache_get(query: str):
    p = os.path.join(CACHE_DIR, _cache_key(query) + ".txt")
    if os.path.exists(p):
        with open(p, "r") as f:
            return f.read()
    return None


def _cache_set(query: str, result: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, _cache_key(query) + ".txt")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        f.write(result)
    os.replace(tmp, p)   # atomic on POSIX — safe for concurrent writers


def ddg_search(query: str, max_results: int = _MAX, sleep: float = _SLEEP) -> str:
    """Query DuckDuckGo and return a plain-text snippet string.

    Cache hit: returns immediately (no network, no sleep).
    Cache miss: sleeps `sleep` seconds, fetches, caches result, returns.
    Falls back to _FALLBACK string on any exception or empty result.
    """
    cached = _cache_get(query)
    if cached is not None:
        return cached

    time.sleep(sleep)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"  [DDG] error for {query!r}: {e}", flush=True)
        hits = []

    if not hits:
        result = _FALLBACK
    else:
        result = "\n\n".join(f"Title: {r['title']}\n{r['body']}" for r in hits)

    _cache_set(query, result)
    return result

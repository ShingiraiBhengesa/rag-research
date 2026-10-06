"""Bounded-concurrency throughput/latency probe for one vLLM endpoint.

Run before scaling, and before deciding whether the attacker and a victim may
share one served process. Measures median/p95 latency and completed
requests/sec at a few concurrency levels using a fixed, harmless prompt. Does
not touch pilot data.

Usage:
    python -m attack.adaptive_pilot.nrp.measure_throughput \
        --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-14B-Instruct
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

PROMPT = "Summarize in one sentence: a short neutral paragraph describing a generic informational passage."


def _one_request(base_url: str, model: str, max_tokens: int, timeout: float) -> float:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return time.perf_counter() - t0


def probe(base_url: str, model: str, concurrency: int, n: int, max_tokens: int, timeout: float) -> dict:
    latencies: list[float] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for lat in pool.map(lambda _: _one_request(base_url, model, max_tokens, timeout), range(n)):
            latencies.append(lat)
    wall = time.perf_counter() - started
    latencies.sort()
    return {
        "concurrency": concurrency, "n": n, "wall_seconds": round(wall, 2),
        "throughput_req_per_s": round(n / wall, 3) if wall else None,
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--n-per-level", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    for c in args.concurrencies:
        print(json.dumps(probe(args.base_url, args.model, c, args.n_per_level, args.max_tokens, args.timeout)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import math
import os
import time
from typing import Callable, Dict, Optional

CACHE_FILENAME = ".auto_chunk_cache.json"


def _load_cache() -> Dict:
    try:
        with open(CACHE_FILENAME, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(d: Dict) -> None:
    try:
        with open(CACHE_FILENAME, "w", encoding="utf8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def calibrate(
    sample_fn: Callable[[int], None],
    sample_n: int = 2000,
    repeats: int = 3,
    warmup: bool = True,
) -> float:
    """Estimate seconds per item using a small benchmark."""
    if warmup:
        try:
            sample_fn(max(1, min(100, sample_n // 10)))
        except Exception:
            pass

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sample_fn(sample_n)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times) if times else 0.0
    return avg / max(1, sample_n)


def suggest_workers_and_chunksize(
    n: int,
    *,
    cpu: Optional[int] = None,
    max_workers: Optional[int] = None,
    sample_cost_per_item: Optional[float] = None,
    desired_chunk_time: float = 0.15,
    min_chunk_time: float = 0.05,
    max_chunk_time: float = 0.6,
    cache_key: Optional[str] = None,
    use_cache: bool = True,
) -> Dict:
    """Suggest (workers, chunksize) for efficient parallel processing."""
    if n <= 0:
        return {"workers": 1, "chunksize": 1, "est_chunk_time": 0.0, "note": "empty input"}

    cpu_count = cpu or (os.cpu_count() or 1)
    if max_workers is None:
        max_workers = cpu_count

    cache = _load_cache() if use_cache else {}
    if sample_cost_per_item is None and cache_key and use_cache:
        entry = cache.get(cache_key)
        if entry and "sample_cost_per_item" in entry:
            sample_cost_per_item = entry["sample_cost_per_item"]

    if sample_cost_per_item is None:
        sample_cost_per_item = 1e-5

    raw_chunksize = max(1, int(desired_chunk_time / sample_cost_per_item))
    min_chunksize = max(1, int(min_chunk_time / sample_cost_per_item))
    max_chunksize = max(1, int(max_chunk_time / sample_cost_per_item))
    chunksize = max(min_chunksize, min(raw_chunksize, max_chunksize))
    chunksize = min(chunksize, n)

    # Worker decision
    if sample_cost_per_item >= 1e-3:
        workers = max(1, cpu_count // 2)
        note = "heavy per-item: reduced workers"
    elif sample_cost_per_item >= 1e-4:
        workers = max(1, (cpu_count * 3) // 4)
        note = "moderate cost"
    else:
        workers = cpu_count
        note = "light workload"

    target_chunks_per_worker = 3
    target_chunks = workers * target_chunks_per_worker
    suggested_chunksize = max(1, math.ceil(n / max(1, target_chunks)))
    final_chunksize = min(n, max(chunksize, suggested_chunksize))

    est_chunk_time = final_chunksize * sample_cost_per_item

    result = {
        "workers": int(workers),
        "chunksize": int(final_chunksize),
        "est_chunk_time": float(est_chunk_time),
        "sample_cost_per_item": float(sample_cost_per_item),
        "cpu_count": int(cpu_count),
        "note": note,
    }

    if cache_key and use_cache:
        cache[cache_key] = {
            "sample_cost_per_item": sample_cost_per_item,
            "cpu_count": cpu_count,
            "updated_at": time.time(),
        }
        _save_cache(cache)

    return result


def calibrate_and_suggest(
    sample_fn: Callable[[int], None],
    n: int,
    *,
    sample_n: int = 2000,
    repeats: int = 3,
    cache_key: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Run calibration and return the best config."""
    cost = calibrate(sample_fn, sample_n, repeats)
    return suggest_workers_and_chunksize(n, sample_cost_per_item=cost, cache_key=cache_key, **kwargs)


if __name__ == "__main__":
    # Example dummy calibration
    def dummy_work(count: int):
        s = 0
        for i in range(count):
            s += (i * 7) % 13
        if s < 0:
            print(s)

    n = 1_000_000
    print("[test] Calibrating dummy workload...")
    cost = calibrate(dummy_work, sample_n=20000, repeats=2)
    print("[test] Estimated cost per item (seconds):", cost)
    cfg = suggest_workers_and_chunksize(n, sample_cost_per_item=cost, cache_key="dummy_test")
    print("[test] Suggested config:", cfg)

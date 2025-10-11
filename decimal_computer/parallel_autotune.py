# decimal_computer/parallel_autotune.py
from __future__ import annotations
import json
import math
import os
import time
from typing import Sequence, Tuple, Union
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer.vector_ops import _is_sequence, ensure_fastdec
from decimal_computer.parallel_pyworkers import mul_vectors_parallel_py
from decimal_computer.context import get_context

Scalar = Union[int, float, str, FastDecimal]
_CACHE_PATH = ".autotune_cache.json"


def _default_candidates(cpu: int, n: int) -> list[Tuple[int, int]]:
    # sensible grid inspired by earlier experiments
    workers_list = sorted({1, max(1, cpu // 2), cpu})
    chunks_list = sorted({
        max(1, n),
        max(1, n // 2),
        max(1, n // 4),
        max(1, n // 8),
        max(1, n // 16),
        max(1, max(1, n // (cpu * 4))),
        max(1, max(1, n // (cpu * 8)))
    })
    # produce a small cross-product but keep it bounded
    candidates = []
    for w in workers_list:
        for c in chunks_list:
            candidates.append((w, c))
    # add a few extra reasonable middlepoints
    mid = max(1, n // (cpu * 2))
    candidates.append((max(1, cpu // 2), mid))
    candidates.append((cpu, max(1, n // (cpu * 6))))
    # unique-preserve
    seen = set()
    out = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict):
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def mul_vectors_parallel_auto(a_list: Sequence[Scalar],
                              b_list: Union[Sequence[Scalar], Scalar],
                              *,
                              sample_size: int = 20_000,
                              force_rebenchmark: bool = False,
                              verbose: bool = True) -> "array":
    """
    Autotune driver that picks (workers,chunksize) then calls mul_vectors_parallel_py.
    - Caches best config in .autotune_cache.json keyed by (cpu, n).
    - Returns array('q') results from mul_vectors_parallel_py using the chosen config.
    """
    if not _is_sequence(b_list):
        b_seq = [b_list] * len(a_list)
    else:
        b_seq = list(b_list)
        if len(b_seq) != len(a_list):
            raise ValueError("a_list and b_list must have same length")

    n = len(a_list)
    if n == 0:
        from array import array
        return array("q")

    cpu = os.cpu_count() or 1
    key = f"cpu{cpu}_n{n}"

    cache = _load_cache()
    if key in cache and not force_rebenchmark:
        cfg = cache[key]
        if verbose:
            print(f"[auto] using cached config workers={cfg['workers']} chunksize={cfg['chunksize']} for n={n} cpu={cpu}")
        return mul_vectors_parallel_py(a_list, b_seq, chunksize=cfg["chunksize"], max_workers=cfg["workers"])

    # build candidates and sample inputs
    candidates = _default_candidates(cpu, n)
    # limit number of candidates for speed
    if len(candidates) > 12:
        candidates = candidates[:12]

    # prepare sample slices (avoid copying full FastDecimal conversion cost)
    sample_n = min(sample_size, n)
    a_sample = a_list[:sample_n]
    b_sample = b_seq[:sample_n]

    best = None
    best_time = float("inf")

    if verbose:
        print(f"[auto] cpu={cpu} n={n} evaluating {len(candidates)} candidates (workers,chunksize) using sample size={sample_n}")

    for workers, chunksize in candidates:
        # run 2 trials and take average
        trials = 2
        times = []
        if verbose:
            print(f"[main] submitting sample run workers={workers} chunksize={chunksize}")
        for _ in range(trials):
            t0 = time.perf_counter()
            try:
                mul_vectors_parallel_py(a_sample, b_sample, chunksize=chunksize, max_workers=workers)
            except Exception:
                # any failure -> treat as slow
                times.append(9999.0)
                continue
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        if verbose:
            print(f"[auto] cand workers={workers} chunksize={chunksize:7} -> avg {avg:.4f}s")
        if avg < best_time:
            best_time = avg
            best = (workers, chunksize)

    # fallback to single-worker if nothing found
    if best is None:
        best = (1, max(1, n // 4))
        best_time = 0.0

    cfg = {"workers": best[0], "chunksize": best[1]}
    cache[key] = cfg
    _save_cache(cache)
    if verbose:
        print(f"[auto] chosen workers={cfg['workers']} chunksize={cfg['chunksize']} (est {best_time:.4f}s)")

    # final run on full data
    return mul_vectors_parallel_py(a_list, b_seq, chunksize=cfg["chunksize"], max_workers=cfg["workers"])

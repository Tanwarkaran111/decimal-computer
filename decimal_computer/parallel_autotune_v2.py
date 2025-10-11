from __future__ import annotations
import json
import math
import os
import time
from typing import Iterable, Tuple

from .fastdecimal import FastDecimal
from .parallel_pyworkers import mul_vectors_parallel_py
from .auto_chunk_optimizer import calibrate_and_suggest, suggest_workers_and_chunksize

CACHE_PATH = ".autotune_cache.json"


def _cpu_count() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except Exception:
        return os.cpu_count() or 1


def _make_key(cpu: int, n: int) -> str:
    return f"cpu{cpu}_n{n}"


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, "r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_cache(d: dict) -> None:
    try:
        with open(CACHE_PATH, "w") as fh:
            json.dump(d, fh, indent=2)
    except Exception:
        pass

def mul_vectors_parallel_auto_v2(
    a_list,
    b_list,
    *,
    sample_size: int = 20000,
    force_rebenchmark: bool = False,
    verbose: bool = True,
    fast_start: bool = True,
):
    n = len(a_list)
    cpu = _cpu_count()
    key = _make_key(cpu, n)

    cache = _load_cache()
    if key in cache and not force_rebenchmark:
        cfg = cache[key]
        if verbose:
            print(f"[auto] using cached config workers={cfg['workers']} chunksize={cfg['chunksize']} for n={n} cpu={cpu}")
        return mul_vectors_parallel_py(a_list, b_list, chunksize=cfg["chunksize"], max_workers=cfg["workers"])

    # Small array shortcut
    if n <= 1000:
        chosen = {"workers": 1, "chunksize": max(1, n)}
        cache[key] = chosen
        _save_cache(cache)
        if verbose:
            print(f"[auto] tiny n={n}, choosing single-worker chunksize={chosen['chunksize']}")
        return mul_vectors_parallel_py(a_list, b_list, chunksize=chosen["chunksize"], max_workers=chosen["workers"])

    # ⚡ Optimizer Path
    if fast_start:
        if verbose:
            print(f"[auto] cpu={cpu} n={n} running optimizer (sample_size={sample_size})")
        try:
            info = None
            # make sure calibrate_and_suggest is callable
            if callable(calibrate_and_suggest):
                info = calibrate_and_suggest(a_list, b_list, cpu_count=cpu, sample_size=sample_size, verbose=verbose)
            else:
                if verbose:
                    print("[auto] calibrate_and_suggest is not callable — skipping optimizer")

            if info:
                # make sure suggest_workers_and_chunksize is callable
                if callable(suggest_workers_and_chunksize):
                    suggested = suggest_workers_and_chunksize(info, n)
                elif isinstance(suggest_workers_and_chunksize, dict):
                    suggested = suggest_workers_and_chunksize
                else:
                    suggested = None

                if suggested:
                    chosen = {"workers": int(suggested["workers"]), "chunksize": int(suggested["chunksize"])}
                    cache[key] = chosen
                    _save_cache(cache)
                    if verbose:
                        est_chunk_time = info.get("est_chunk_time", None)
                        cost_per_item = info.get("cost_per_item", None)
                        print(f"[auto] optimizer suggested workers={chosen['workers']} chunksize={chosen['chunksize']}"
                              f"{'' if est_chunk_time is None else f' (est_chunk_time={est_chunk_time:.3f}s, cost_per_item={cost_per_item:.2e})'}")

                    # Execute with chosen config
                    t0 = time.perf_counter()
                    out = mul_vectors_parallel_py(a_list, b_list,
                                                  chunksize=chosen["chunksize"],
                                                  max_workers=chosen["workers"])
                    elapsed = time.perf_counter() - t0
                    if verbose:
                        print(f"[main] submitting {math.ceil(n / chosen['chunksize'])} chunk(s) "
                              f"using up to {chosen['workers']} worker(s) on {cpu} CPU(s) "
                              f"(chunksize≈{chosen['chunksize']})")
                        print(f"[main] parallel multiply completed in {elapsed:.3f}s")
                    return out
                else:
                    if verbose:
                        print("[auto] optimizer returned no valid suggestion, falling back to full autotune")
            else:
                if verbose:
                    print("[auto] optimizer returned no info, falling back to full autotune")

        except Exception as e:
            if verbose:
                print(f"[auto] optimizer raised, falling back to full autotune: {e}")

    # 🧠 Fallback full search
    def _generate_candidates(cpu: int, n: int):
        workers_list = [1]
        if cpu >= 2:
            workers_list += [max(1, cpu // 2), cpu]
        workers_list = sorted(set(workers_list))

        chunks = []
        chunks.append(max(1, n))
        for k in (2, 4, 8, 16, 32, 64, 128):
            c = max(1, n // k)
            if c not in chunks:
                chunks.append(c)
        for c in (12500, 25000, 50000, 8333, 16667, 33333, 4166, 2083):
            if 0 < c <= n and c not in chunks:
                chunks.append(c)

        for w in workers_list:
            for c in chunks:
                yield (w, c)

    def _time_candidate(workers: int, chunksize: int, sample_size_local: int) -> float:
        a = [FastDecimal.from_str("1.23")] * sample_size_local
        b = [FastDecimal.from_str("4.56")] * sample_size_local
        t0 = time.perf_counter()
        mul_vectors_parallel_py(a, b, chunksize=chunksize, max_workers=workers)
        return time.perf_counter() - t0

    candidates = list(_generate_candidates(cpu, n))
    if verbose:
        print(f"[auto] cpu={cpu} n={n} evaluating {len(candidates)} candidates (workers,chunksize) using sample size={sample_size}")

    best = None
    best_t = float("inf")
    sample_n = min(sample_size, max(1, n))
    for workers, chunksize in candidates:
        try:
            avg = _time_candidate(workers, chunksize, sample_n)
        except Exception:
            avg = float("inf")
        if verbose:
            print(f"[auto] cand workers={workers} chunksize={str(chunksize).rjust(6)} -> avg {avg:.4f}s")
        if avg < best_t:
            best_t = avg
            best = (workers, chunksize)

    chosen = {"workers": 1, "chunksize": max(1, n)} if best is None else {"workers": int(best[0]), "chunksize": int(best[1])}
    cache[key] = chosen
    _save_cache(cache)
    if verbose:
        print(f"[auto] chosen workers={chosen['workers']} chunksize={chosen['chunksize']} (est {best_t:.4f}s)")

    t0 = time.perf_counter()
    out = mul_vectors_parallel_py(a_list, b_list, chunksize=chosen["chunksize"], max_workers=chosen["workers"])
    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"[main] submitting {math.ceil(n / chosen['chunksize'])} chunk(s) using up to {chosen['workers']} worker(s) "
              f"on {cpu} CPU(s) (chunksize={chosen['chunksize']})")
        print(f"[main] parallel multiply completed in {elapsed:.3f}s")
    return out

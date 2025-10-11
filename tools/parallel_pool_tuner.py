from __future__ import annotations
import time
import json
import os
from typing import Iterable, Tuple
from decimal_computer.fastdecimal import FastDecimal
from tools.parallel_pool_shared_output import mul_vectors_pool_shared_output

def candidates(cpu: int, n: int) -> Iterable[Tuple[int,int]]:
    # reasonable candidate grid: workers in {1, cpu//2, cpu}, chunksize in {n//(w*4)} scaled
    ws = sorted(set([1, max(1, cpu//2), cpu]))
    cand = []
    for w in ws:
        # base chunksize so each worker gets ~4 sub-tasks
        base = max(1, n // max(1, w * 4))
        # try base and some multiples/divisors
        for s in [base//4, base//2, base, base*2, base*4]:
            if s < 1:
                continue
            cand.append((w, int(s)))
    # unique and sensible ordering
    seen = set()
    out = []
    for w,s in cand:
        if (w,s) not in seen:
            out.append((w,s))
            seen.add((w,s))
    return out

def time_trial(a, b, workers, chunksize, repeats=2) -> float:
    # run the call 'repeats' times and return average elapsed
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mul_vectors_pool_shared_output(a, b, chunksize=chunksize, max_workers=workers)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

def run(n: int = 200_000, sample_repeats: int = 2, save_cache: bool = True):
    cpu = os.cpu_count() or 1
    a = [FastDecimal.from_str('1.23')] * n
    b = [FastDecimal.from_str('4.56')] * n

    cand = list(candidates(cpu, n))
    print(f"[tuner] cpu={cpu} n={n} evaluating {len(cand)} candidates (workers,chunksize) using sample size={n//10}")
    results = []
    for (w,s) in cand:
        # time a smaller sample first (faster feedback)
        sample_n = max(1, min(n//10, s*max(1, w)))
        # Build sample inputs
        a_sample = a[:sample_n]
        b_sample = b[:sample_n]
        # Quick sanity run to warm caches / ensure correctness
        _ = mul_vectors_pool_shared_output(a_sample, b_sample, chunksize=s, max_workers=w)
        avg = time_trial(a, b, workers=w, chunksize=s, repeats=sample_repeats)
        results.append(((w,s), avg))
        print(f"[auto] cand workers={w:2d} chunksize={s:8d} -> avg {avg:.4f}s")

    # pick best (min avg)
    best = min(results, key=lambda rs: rs[1])
    (best_w, best_s), best_t = best
    print(f"[auto] chosen workers={best_w} chunksize={best_s} (est {best_t:.4f}s)")

    cache = {
        "cpu": cpu,
        "n": n,
        "workers": best_w,
        "chunksize": best_s,
        "time_est": best_t,
        "candidates": [(w,s,t) for ((w,s),t) in results],
    }
    if save_cache:
        with open(".parallel_pool_tune.json", "w") as fh:
            json.dump(cache, fh)
        print("[auto] cached config written to .parallel_pool_tune.json")
    return cache

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    run(n=args.n, sample_repeats=args.repeats)

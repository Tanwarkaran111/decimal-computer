# phase3/tune_blocksize.py
"""
Auto-tune block_size for blocked schoolbook (_multiply_block_rows).
Outputs:
 - phase3_blocksize_table.json  (best block per n)
 - phase3_blocksize_results.csv (detailed timings)
"""
from __future__ import annotations
import time
import random
import statistics
import json
import csv
from pathlib import Path
from typing import List, Dict

# Import your blocked multiply (make sure name matches your code)
from phase2.parallel_karatsuba import _multiply_block_rows as blocked

# Config
NS = [64, 128, 256, 512]          # sizes to tune
BLOCK_SIZES = [8, 16, 32, 48, 64] # block sizes to try
TRIALS = 3
OUT_JSON = Path("phase3_blocksize_table.json")
OUT_CSV = Path("phase3_blocksize_results.csv")
RNG_SEED = 123456

def gen_matrix(n: int, seed: int) -> List[List[int]]:
    r = random.Random(seed)
    return [[r.randint(0, 9) for _ in range(n)] for __ in range(n)]

def time_block(n: int, b: int, A: List[List[int]], B: List[List[int]]) -> float:
    t0 = time.perf_counter()
    # blocked accepts block_size as positional third argument or keyword `block_size`
    try:
        blocked(A, B, b)
    except TypeError:
        # fallback: if function signature expects only two args, call as-is
        blocked(A, B)
    return time.perf_counter() - t0

def tune():
    results: Dict[int, Dict[int, List[float]]] = {}
    # Prepare CSV
    with OUT_CSV.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["n", "block_size", "trial", "elapsed_seconds"])

        for n in NS:
            # generate deterministic matrices per n for fairness
            seed = RNG_SEED + n
            A = gen_matrix(n, seed)
            B = gen_matrix(n, seed + 1)
            results[n] = {}
            for b in BLOCK_SIZES:
                times: List[float] = []
                # warmup single call
                try:
                    blocked(A, B, b)
                except TypeError:
                    try:
                        blocked(A, B)
                    except Exception:
                        # If blocked fails for (n,b), record large times so it won't be selected
                        times = [float("inf")] * TRIALS
                        for t in range(TRIALS):
                            writer.writerow([n, b, t + 1, "inf"])
                        results[n][b] = times
                        continue

                for t in range(TRIALS):
                    elapsed = time_block(n, b, A, B)
                    times.append(elapsed)
                    writer.writerow([n, b, t + 1, f"{elapsed:.6f}"])
                    fcsv.flush()
                results[n][b] = times

    # Choose best block size per n (lowest mean)
    best_map: Dict[int, int] = {}
    for n, bmap in results.items():
        best_b = None
        best_mean = float("inf")
        for b, times in bmap.items():
            # ignore infinite rounds
            valid_times = [x for x in times if x != float("inf")]
            if not valid_times:
                continue
            mean_t = statistics.mean(valid_times)
            if mean_t < best_mean:
                best_mean = mean_t
                best_b = b
        if best_b is None:
            best_b = BLOCK_SIZES[0]  # fallback
        best_map[n] = best_b

    # Save JSON
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"best_block_for_n": best_map, "ns": NS, "block_sizes": BLOCK_SIZES, "trials": TRIALS}, f, indent=2)

    # Print summary
    print("Tuning complete. Summary:")
    for n in NS:
        chosen = best_map.get(n)
        means = {b: (statistics.mean([x for x in results[n][b] if x != float("inf")]) if any(x != float("inf") for x in results[n][b]) else float("inf")) for b in BLOCK_SIZES}
        print(f" n={n}: best_block={chosen}, means={', '.join([f'{b}:{means[b]:.3f}' if means[b] != float('inf') else f'{b}:inf' for b in BLOCK_SIZES])}")

    print(f"\nWrote JSON -> {OUT_JSON}")
    print(f"Wrote CSV  -> {OUT_CSV}")

if __name__ == "__main__":
    tune()

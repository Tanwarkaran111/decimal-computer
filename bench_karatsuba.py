# bench_karatsuba.py
# Usage: python bench_karatsuba.py
# Produces averaged timings and writes bench_results.csv

import time
import random
import statistics
import csv
from phase8_karatsuba_nogil import multiply

SIZES = [2048, 4096, 8192, 16384, 32768, 65536]  # bits
REPEATS = 7
WARMUP = 2

def time_pair(bits, repeats=REPEATS):
    a = random.getrandbits(bits)
    b = random.getrandbits(bits)
    # warmup
    for _ in range(WARMUP):
        multiply(a, b)
        _ = a * b
    kar_times = []
    builtin_times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); multiply(a, b); kar_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _ = a * b; builtin_times.append(time.perf_counter() - t0)
    return statistics.mean(kar_times), statistics.mean(builtin_times), statistics.stdev(kar_times), statistics.stdev(builtin_times)

def main():
    print("Running benchmark. This may take a while for large sizes...")
    rows = []
    for bits in SIZES:
        km, bm, kstd, bstd = time_pair(bits)
        ratio = km / bm if bm > 0 else float("inf")
        print(f"{bits:6d} bits: kar={km:.6f}s ±{kstd:.6f}  builtin={bm:.6f}s ±{bstd:.6f}  ratio={ratio:.3f}")
        rows.append({"bits": bits, "kar_mean": km, "kar_std": kstd, "builtin_mean": bm, "builtin_std": bstd, "ratio": ratio})
    # write csv
    with open("bench_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nResults written to bench_results.csv")

if __name__ == "__main__":
    main()

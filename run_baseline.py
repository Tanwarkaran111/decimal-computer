# run_baseline.py
import time
import numpy as np
from pathlib import Path
import csv

OUT = Path("baseline_results.csv")

def run_once(n, dtype=np.float32, repeats=3):
    # Use contiguous, well-aligned arrays
    A = np.random.randn(n, n).astype(dtype)
    B = np.random.randn(n, n).astype(dtype)
    # warmup
    np.dot(A, B)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = sum(times) / len(times)
    gflops = (2.0 * n * n * n) / (mean * 1e9)  # approximate FLOPs for GEMM
    return mean, gflops

def main():
    sizes = [2048, 4096]  # baseline sizes (you can add 1024 to test locally)
    rows = []
    for n in sizes:
        mean, gflops = run_once(n, repeats=3)
        print(f"n={n}: mean={mean:.4f}s, gflops={gflops:.2f}")
        rows.append({"n": n, "mean_s": mean, "gflops": gflops})
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n", "mean_s", "gflops"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", OUT)

if __name__ == "__main__":
    main()

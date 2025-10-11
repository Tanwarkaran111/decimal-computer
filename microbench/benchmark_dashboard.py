# microbench/benchmark_dashboard.py
import csv
import time
import os
from array import array
from decimal_computer import DecimalContext, FastDecimal
from decimal_computer import vector_ops
from decimal_computer.parallel import mul_vectors_parallel
import matplotlib.pyplot as plt

OUT_DIR = "microbench/results"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "benchmark_results.csv")

SIZES = [20_000, 200_000, 2_000_000, 5_000_000, 10_000_000]  # adjust for your machine
ROUNDS = 3

def bench_once(n):
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP", mode="auto"):
        a = [FastDecimal.from_str("1.2345")] * n
        b = [FastDecimal.from_str("2.5")] * n

        # baseline vector_ops.mul_vectors (auto)
        t0 = time.perf_counter()
        _ = vector_ops.mul_vectors(a, b)
        t1 = time.perf_counter()
        t_safe = t1 - t0

        # frozen baseline
        t0 = time.perf_counter()
        _ = vector_ops.mul_vectors_frozen(a, b)
        t1 = time.perf_counter()
        t_frozen = t1 - t0

        # parallel raw path (array result)
        t0 = time.perf_counter()
        res_arr = mul_vectors_parallel(a, b, chunksize=200_000)
        t1 = time.perf_counter()
        t_parallel = t1 - t0

        # fast view path (same as mul_vectors_fast returning view)
        t0 = time.perf_counter()
        out = vector_ops.mul_vectors_fast(a, b)
        t1 = time.perf_counter()
        t_fast = t1 - t0

    return t_safe, t_frozen, t_parallel, t_fast, len(res_arr)

def run_bench():
    rows = []
    for n in SIZES:
        for r in range(ROUNDS):
            print(f"Benchmarking N={n:,} round {r+1}/{ROUNDS}")
            times = bench_once(n)
            rows.append((n,) + times)
            # write intermediate results to CSV
            with open(CSV_PATH, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow((n,) + times)
    # Plot summary
    sizes = sorted(set(r[0] for r in rows))
    safe_mean = []
    frozen_mean = []
    parallel_mean = []
    fast_mean = []
    for s in sizes:
        srows = [r for r in rows if r[0] == s]
        safe_mean.append(sum(r[1] for r in srows)/len(srows))
        frozen_mean.append(sum(r[2] for r in srows)/len(srows))
        parallel_mean.append(sum(r[3] for r in srows)/len(srows))
        fast_mean.append(sum(r[4] for r in srows)/len(srows))

    plt.figure(figsize=(10,6))
    plt.plot(sizes, safe_mean, label="mul_vectors (auto/safe)")
    plt.plot(sizes, frozen_mean, label="mul_vectors_frozen")
    plt.plot(sizes, parallel_mean, label="mul_vectors_parallel (processes)")
    plt.plot(sizes, fast_mean, label="mul_vectors_fast (view/auto)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N (log scale)")
    plt.ylabel("Time seconds (log scale)")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUT_DIR, "benchmark_plot.png")
    plt.savefig(plot_path)
    print("Saved plot to:", plot_path)
    print("CSV written to:", CSV_PATH)

if __name__ == "__main__":
    # write header if missing
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(("N", "mul_vectors", "mul_vectors_frozen", "mul_vectors_parallel", "mul_vectors_fast", "res_len"))
    run_bench()

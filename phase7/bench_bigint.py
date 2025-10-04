# phase7/bench_bigint.py
import argparse
import csv
import time
import random
import math
from pathlib import Path

from phase7.bigint import BigInt

try:
    import gmpy2
    HAVE_GMPY2 = True
except Exception:
    HAVE_GMPY2 = False

def rand_decimal(digits: int, rng=random.Random()):
    return int("".join(str(rng.randint(0, 9)) for _ in range(digits)))

def bench_sizes(sizes, trials=3, out_csv="phase7_bench_bigint.csv", plot=False, plot_out="phase7_bench_bigint.png"):
    rows = []
    rng = random.Random(12345)
    for d in sizes:
        print(f"\n=== digits={d} ===")
        a = rand_decimal(d, rng)
        b = rand_decimal(d, rng)

        # Python int baseline
        t_list = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = a * b
            t1 = time.perf_counter()
            t_list.append(t1 - t0)
        py_median = sorted(t_list)[len(t_list)//2]
        print(f"Python int median: {py_median:.6f}s")

        # BigInt (uses gmpy2 if installed)
        A, B = BigInt.from_int(a), BigInt.from_int(b)
        t_list = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = A * B
            t1 = time.perf_counter()
            t_list.append(t1 - t0)
        big_median = sorted(t_list)[len(t_list)//2]
        print(f"BigInt median: {big_median:.6f}s")

        rows.append((d, py_median, big_median))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["digits", "python_int_s", "bigint_s"])
        for r in rows:
            w.writerow(r)
    print(f"Results written to {out_csv}")

    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot")
            return
        digits = [r[0] for r in rows]
        py = [r[1] for r in rows]
        bi = [r[2] for r in rows]
        plt.figure()
        plt.loglog(digits, py, marker="o", label="python int")
        plt.loglog(digits, bi, marker="o", label="BigInt")
        plt.xlabel("decimal digits")
        plt.ylabel("median time (s)")
        plt.title("phase7: BigInt multiply vs Python int")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(plot_out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Plot saved to {plot_out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[1000,5000,20000,50000,100000])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", type=str, default="phase7_bench_bigint.csv")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-out", type=str, default="phase7_bench_bigint.png")
    args = p.parse_args()
    bench_sizes(args.sizes, trials=args.trials, out_csv=args.out, plot=args.plot, plot_out=args.plot_out)

if __name__ == "__main__":
    main()

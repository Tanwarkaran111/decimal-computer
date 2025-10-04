# experiments/benchmark_vs_numpy.py
"""
Compare NumPy dot() vs decimal GEMM auto-runtime.

Usage examples:
  python experiments/benchmark_vs_numpy.py --sizes 128 256 --repeats 5
  python experiments/benchmark_vs_numpy.py --sizes 512 --repeats 3 --digits 8
"""
from pathlib import Path
import time
import json
import argparse
import csv
import platform

import numpy as np

OUT = Path("experiments/benchmark_vs_numpy.csv")
OUT.parent.mkdir(exist_ok=True)

def run_numpy(n, repeats):
    A = np.random.randn(n, n).astype(np.float64)
    B = np.random.randn(n, n).astype(np.float64)
    # warmup
    _ = A @ B
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = float(sum(times) / len(times))
    std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    return {"mean_s": mean, "std_s": std, "samples": None, "repeat_times": json.dumps(times)}

def run_decimal_auto(n, digits, repeats, verbose=False):
    """
    Try to import decimal_gemm_auto and time repeated calls.
    Returns dict with mean/std (or None on failure) and last error/log.
    """
    try:
        from decimal_computer.auto_runtime import decimal_gemm_auto
    except Exception as e:
        return {"mean_s": None, "std_s": None, "samples": None, "repeat_times": "", "error": f"import error: {e}"}

    # Build tiny digit matrices in case decimal_gemm_auto expects digit-lists
    try:
        from decimal_computer.decimal_digit_starter import int_to_digits
        A = [[int_to_digits(1, digits) for _ in range(n)] for _ in range(n)]
        B = [[int_to_digits(2, digits) for _ in range(n)] for _ in range(n)]
        use_digits_repr = True
    except Exception:
        # fallback to int matrices (some implementations accept plain ints)
        A = [[1 for _ in range(n)] for _ in range(n)]
        B = [[2 for _ in range(n)] for _ in range(n)]
        use_digits_repr = False

    times = []
    last_err = None
    # warmup single call (do not include timing)
    try:
        decimal_gemm_auto(A, B, size=n, digits=digits, verbose=(verbose))
    except Exception:
        pass

    for i in range(repeats):
        t0 = time.perf_counter()
        try:
            _ = decimal_gemm_auto(A, B, size=n, digits=digits, verbose=(verbose))
            t1 = time.perf_counter()
            times.append(t1 - t0)
        except Exception as e:
            # stop on failure but record error
            last_err = str(e)
            break

    if not times:
        return {"mean_s": None, "std_s": None, "samples": None, "repeat_times": "", "error": last_err or "failed without exception"}

    mean = float(sum(times) / len(times))
    std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    return {"mean_s": mean, "std_s": std, "samples": None, "repeat_times": json.dumps(times), "error": None}

def append_rows(path, rows, fieldnames):
    first = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if first:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--digits", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rows = []
    for n in args.sizes:
        print(f"Benchmarking n={n} (digits={args.digits}) ...")
        # NumPy baseline
        np_res = run_numpy(n, args.repeats)
        print(f"  NumPy mean={np_res['mean_s']:.6f}s std={np_res['std_s']:.6f}s")

        # Decimal auto runtime
        dec_res = run_decimal_auto(n, args.digits, args.repeats, verbose=args.verbose)
        if dec_res.get("mean_s") is None:
            print(f"  Decimal run failed for n={n}: {dec_res.get('error')}")
        else:
            print(f"  Decimal mean={dec_res['mean_s']:.6f}s std={dec_res['std_s']:.6f}s")

        # compute speedup if possible
        speedup = None
        if np_res["mean_s"] is not None and dec_res.get("mean_s") is not None:
            try:
                speedup = float(np_res["mean_s"]) / float(dec_res["mean_s"])
            except Exception:
                speedup = None

        row = {
            "timestamp": time.time(),
            "machine": platform.node(),
            "n": n,
            "digits": args.digits,
            "numpy_mean_s": np_res["mean_s"],
            "numpy_std_s": np_res["std_s"],
            "decimal_mean_s": dec_res.get("mean_s"),
            "decimal_std_s": dec_res.get("std_s"),
            "speedup": speedup,
            "decimal_error": dec_res.get("error"),
            "decimal_repeat_times": dec_res.get("repeat_times"),
            "numpy_repeat_times": np_res.get("repeat_times"),
        }
        rows.append(row)

    fieldnames = ["timestamp","machine","n","digits","numpy_mean_s","numpy_std_s","decimal_mean_s","decimal_std_s","speedup","decimal_error","decimal_repeat_times","numpy_repeat_times"]
    append_rows(OUT, rows, fieldnames)
    print("Saved CSV ->", OUT)

if __name__ == "__main__":
    main()

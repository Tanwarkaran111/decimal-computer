# phase4/bench_fft.py
"""
Benchmark runner for Phase 4: compare FFT-based integer multiplication
(phase4.fft_multiply.multiply_ints) against schoolbook and Karatsuba
implementations (pure Python).

Usage:
    python -m phase4.bench_fft         # runs default sizes/trials
    python -m phase4.bench_fft --plot   # also generates a log-log PNG plot

Results are printed to stdout and saved as CSV to phase4/bench_fft_results.csv
If --plot is passed, a PNG is saved to phase4/bench_fft_plot.png

Notes:
- This script is intentionally self-contained so it doesn't rely on Phase 3 module names.
- Default digit sizes are conservative; increase if you want heavier stress tests.
"""

import argparse
import csv
import math
import random
import time
import gc
import sys
from typing import Tuple

# Import FFT multiply from this package (phase4)
from .fft_multiply import multiply_ints

# ------------------------
# Schoolbook multiplication (base-split convolution)
# ------------------------

def _to_digits(n: int, base: int):
    if n == 0:
        return [0]
    digs = []
    while n:
        digs.append(n % base)
        n //= base
    return digs


def schoolbook_multiply(a: int, b: int, base: int = 10_000) -> int:
    """Grade-school O(n*m) multiplication using digits in given base.
    Base chosen to reduce digit count; result reconstructed exactly.
    """
    sign = -1 if (a < 0) ^ (b < 0) else 1
    a = abs(a)
    b = abs(b)
    da = _to_digits(a, base)
    db = _to_digits(b, base)
    n = len(da)
    m = len(db)
    res = [0] * (n + m)
    for i in range(n):
        ai = da[i]
        for j in range(m):
            res[i + j] += ai * db[j]
    # handle carries
    carry = 0
    for i in range(len(res)):
        total = res[i] + carry
        carry = total // base
        res[i] = total % base
    while carry:
        res.append(carry % base)
        carry //= base
    # convert back
    out = 0
    for d in reversed(res):
        out = out * base + d
    return sign * out

# ------------------------
# Karatsuba multiplication (recursive) - avoids str() for digit count
# ------------------------

def karatsuba_multiply(x: int, y: int) -> int:
    """Pure-python Karatsuba for integers. Falls back to Python * for small sizes.
    This implementation splits by base 10^m where m is half the number of digits.
    """
    # handle sign
    sign = -1 if (x < 0) ^ (y < 0) else 1
    a = abs(x)
    b = abs(y)
    # base case: small integers -> use built-in
    if a < 10**4 or b < 10**4:
        return sign * (a * b)

    # number of decimal digits (avoid str() to bypass Python’s digit-limit guard)
    na = int(a.bit_length() / math.log2(10)) + 1
    nb = int(b.bit_length() / math.log2(10)) + 1
    n = max(na, nb)
    m = n // 2
    base = 10 ** m

    # split
    high1, low1 = divmod(a, base)
    high2, low2 = divmod(b, base)

    z0 = karatsuba_multiply(low1, low2)
    z2 = karatsuba_multiply(high1, high2)
    z1 = karatsuba_multiply(low1 + high1, low2 + high2) - z2 - z0

    return sign * (z2 * (base ** 2) + z1 * base + z0)

# ------------------------
# Utility: random integer generator by decimal digits
# ------------------------

def random_decimal_int(digits: int) -> int:
    if digits <= 0:
        return 0
    lower = 10 ** (digits - 1)
    upper = 10 ** digits - 1
    return random.randint(lower, upper)

# ------------------------
# Benchmark harness
# ------------------------

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

# memory estimator for FFT arrays (conservative)
def estimate_fft_memory_bytes(decimal_digits: int, base: int) -> int:
    # limbs per operand
    digits_per_limb = math.log10(base)
    n_limbs = math.ceil(decimal_digits / digits_per_limb)
    L = _next_power_of_two(2 * n_limbs)
    # estimate: 3 arrays of complex128 (16 bytes each) -> 3 * L * 16 = 48 * L
    return 48 * L

# small helper to format bytes
def _fmt_bytes(b: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}PB"

# Updated time_fn that disables GC for timing
def time_fn(fn, *args, trials: int = 3) -> Tuple[float, list]:
    times = []
    # warmup
    fn(*args)
    # try to reduce GC jitter
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for _ in range(trials):
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            times.append(t1 - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    times.sort()
    median = times[len(times) // 2]
    return median, times


def bench_sizes(sizes, trials=3, base_seed=12345, base=1000, karatsuba_skip_digits=5000):
    random.seed(base_seed)
    rows = []
    for digits in sizes:
        a = random_decimal_int(digits)
        b = random_decimal_int(digits)
        expected = a * b

        mem_est = estimate_fft_memory_bytes(digits, base)
        print(f"\n=== digits={digits} (approx bits={digits*3.3219:.0f}) ===")
        print(f"FFT memory estimate: {_fmt_bytes(mem_est)}  (base={base})")

        # FFT multiply
        t_fft, fft_times = time_fn(multiply_ints, a, b, trials=trials)
        fft_result = multiply_ints(a, b)
        print(f"FFT    median: {t_fft:.6f}s  trials: {fft_times}")
        if fft_result != expected:
            print("ERROR: FFT result mismatch!")

        # Karatsuba: skip if too large
        if digits > karatsuba_skip_digits:
            t_kar = float('inf')
            kar_times = []
            print(f"Karatsuba: skipped for digits>{karatsuba_skip_digits}")
        else:
            t_kar, kar_times = time_fn(karatsuba_multiply, a, b, trials=trials)
            kar_result = karatsuba_multiply(a, b)
            print(f"Karatsuba median: {t_kar:.6f}s  trials: {kar_times}")
            if kar_result != expected:
                print("ERROR: Karatsuba result mismatch!")

        # Schoolbook
        t_sch, sch_times = time_fn(schoolbook_multiply, a, b, trials=trials)
        sch_result = schoolbook_multiply(a, b)
        print(f"Schoolbook median: {t_sch:.6f}s  trials: {sch_times}")
        if sch_result != expected:
            print("ERROR: Schoolbook result mismatch!")

        rows.append({
            "digits": digits,
            "base": base,
            "fft_median_s": t_fft,
            "karatsuba_median_s": t_kar if kar_times else None,
            "schoolbook_median_s": t_sch,
            "fft_mem_est_bytes": mem_est,
        })
    return rows

# ------------------------
# CLI and run
# ------------------------

def _maybe_plot(rows, out_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available; skipping plot. Install matplotlib to enable plotting.")
        return

    digits = [r["digits"] for r in rows]
    fft_times = [r["fft_median_s"] for r in rows]
    kar_times = [r["karatsuba_median_s"] if r["karatsuba_median_s"] is not None else float('nan') for r in rows]
    sch_times = [r["schoolbook_median_s"] for r in rows]

    plt.figure()
    plt.loglog(digits, fft_times, marker='o', label='FFT')
    plt.loglog(digits, kar_times, marker='o', label='Karatsuba')
    plt.loglog(digits, sch_times, marker='o', label='Schoolbook')
    plt.xlabel('Decimal digits')
    plt.ylabel('Median time (s)')
    plt.title('Phase4: Multiplication benchmarks')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FFT vs Karatsuba vs Schoolbook")
    parser.add_argument("--sizes", nargs="*", type=int,
                        default=[10, 50, 200, 800],
                        help="list of decimal-digit sizes to test (default: 10 50 200 800)")
    parser.add_argument("--trials", type=int, default=3, help="trials per test (default 3)")
    parser.add_argument("--out", type=str, default="phase4/bench_fft_results.csv",
                        help="CSV output path")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--base", type=int, default=1000,
                        help="digit base (default 1000, ~3 decimal digits per limb)")
    parser.add_argument("--skip-karatsuba-digits", type=int, default=5000,
                        help="skip karatsuba if decimal-digit size > this (default 5000)")
    parser.add_argument("--plot", action='store_true', help="generate log-log plot PNG")
    parser.add_argument("--plot-out", type=str, default="phase4/bench_fft_plot.png",
                        help="PNG output path for plot")
    parser.add_argument("--bases", nargs="*", type=int,
                        help="list of base values to sweep, e.g. --bases 100 1000 10000")
    parser.add_argument("--combined-plot-out", type=str, default="phase4/bench_fft_bases.png",
                        help="output PNG for combined base-sweep plot")
    ...
    args = parser.parse_args()

    # if bases provided -> run sweep
    if args.bases:
        all_rows = []  # list of (base, rows)
        for base_val in args.bases:
            print(f"--- Running base={base_val} ---")
            rows = bench_sizes(args.sizes,
                               trials=args.trials,
                               base_seed=args.seed,
                               base=base_val,
                               karatsuba_skip_digits=args.skip_karatsuba_digits)
            # save per-base CSV
            out_csv = args.out.replace(".csv", f"_base{base_val}.csv")
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "digits", "base", "fft_median_s", "karatsuba_median_s", "schoolbook_median_s", "fft_mem_est_bytes"
                ])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Results for base={base_val} written to {out_csv}")
            all_rows.append((base_val, rows))

        if args.plot:
            # combined plot: one line per algorithm per base (FFT lines grouped by base)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(9,6))
                for base_val, rows in all_rows:
                    digits = [r["digits"] for r in rows]
                    fft_times = [r["fft_median_s"] for r in rows]
                    sch_times = [r["schoolbook_median_s"] for r in rows]
                    # prefer plotting FFT and Schoolbook; skip karatsuba if missing
                    plt.loglog(digits, fft_times, marker='o', linestyle='-', label=f'FFT (base={base_val})')
                    plt.loglog(digits, sch_times, marker='x', linestyle='--', label=f'Schoolbook (base={base_val})')
                plt.xlabel('Decimal digits')
                plt.ylabel('Median time (s)')
                plt.title('Phase4: Base sweep (FFT vs Schoolbook)')
                plt.grid(True, which='both', ls='--', lw=0.5)
                plt.legend()
                plt.savefig(args.combined_plot_out, bbox_inches='tight', dpi=200)
                plt.close()
                print(f"Combined base-sweep plot saved to {args.combined_plot_out}")
            except Exception:
                print("matplotlib not available; skipping combined plot.")
        # done
        return

    args = parser.parse_args()

    rows = bench_sizes(args.sizes,
                       trials=args.trials,
                       base_seed=args.seed,
                       base=args.base,
                       karatsuba_skip_digits=args.skip_karatsuba_digits)

    # write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "digits", "base", "fft_median_s", "karatsuba_median_s", "schoolbook_median_s", "fft_mem_est_bytes"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nResults written to {args.out}")

    if args.plot:
        _maybe_plot(rows, args.plot_out)


if __name__ == "__main__":
    main()

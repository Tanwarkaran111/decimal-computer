# phase5/bench_ntt.py
"""
Benchmark runner for Phase 5: compare NTT-based integer multiplication
(phase5.ntt_multiply.multiply_ints_ntt) against FFT (phase4.fft_multiply.multiply_ints),
Karatsuba and Schoolbook implementations.

Usage:
    python -m phase5.bench_ntt --sizes 2000 8000 --trials 3 --plot

Outputs:
    - CSV: phase5/bench_ntt_results.csv
    - PNG plot (optional): phase5/bench_ntt_plot.png

This file mirrors the style of phase4/bench_fft.py to keep results comparable.
"""

import argparse
import csv
import math
import random
import time
import gc
from typing import Tuple

# Imports
from .ntt_multiply import multiply_ints_ntt
# import FFT multiply (phase4) if available
try:
    from phase4.fft_multiply import multiply_ints as multiply_ints_fft
    _HAVE_FFT = True
except Exception:
    _HAVE_FFT = False

# Local schoolbook & karatsuba (copied/compatible with phase4 runner)

def _to_digits(n: int, base: int):
    if n == 0:
        return [0]
    digs = []
    while n:
        digs.append(n % base)
        n //= base
    return digs


def schoolbook_multiply(a: int, b: int, base: int = 10_000) -> int:
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
    carry = 0
    for i in range(len(res)):
        total = res[i] + carry
        carry = total // base
        res[i] = total % base
    while carry:
        res.append(carry % base)
        carry //= base
    out = 0
    for d in reversed(res):
        out = out * base + d
    return sign * out


def karatsuba_multiply(x: int, y: int) -> int:
    sign = -1 if (x < 0) ^ (y < 0) else 1
    a = abs(x)
    b = abs(y)
    if a < 10**4 or b < 10**4:
        return sign * (a * b)
    na = int(a.bit_length() / math.log2(10)) + 1
    nb = int(b.bit_length() / math.log2(10)) + 1
    n = max(na, nb)
    m = n // 2
    base = 10 ** m
    high1, low1 = divmod(a, base)
    high2, low2 = divmod(b, base)
    z0 = karatsuba_multiply(low1, low2)
    z2 = karatsuba_multiply(high1, high2)
    z1 = karatsuba_multiply(low1 + high1, low2 + high2) - z2 - z0
    return sign * (z2 * (base ** 2) + z1 * base + z0)

# Utility

def random_decimal_int(digits: int) -> int:
    if digits <= 0:
        return 0
    lower = 10 ** (digits - 1)
    upper = 10 ** digits - 1
    return random.randint(lower, upper)


def time_fn(fn, *args, trials: int = 3) -> Tuple[float, list]:
    times = []
    # warmup
    fn(*args)
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


def bench_sizes(sizes, trials=3, base_seed=12345, base=1000, skip_karatsuba_digits=5000):
    random.seed(base_seed)
    rows = []
    for digits in sizes:
        a = random_decimal_int(digits)
        b = random_decimal_int(digits)
        expected = a * b
        print(f"\n=== digits={digits} (approx bits={digits*3.3219:.0f}) ===")

        # NTT
        t_ntt, nt_t = time_fn(multiply_ints_ntt, a, b, trials=trials)
        got_ntt = multiply_ints_ntt(a, b)
        print(f"NTT    median: {t_ntt:.6f}s  trials: {nt_t}")
        if got_ntt != expected:
            print("ERROR: NTT mismatch!")

        # FFT (if available)
        if _HAVE_FFT:
            t_fft, fft_t = time_fn(multiply_ints_fft, a, b, trials=trials)
            got_fft = multiply_ints_fft(a, b)
            print(f"FFT    median: {t_fft:.6f}s  trials: {fft_t}")
            if got_fft != expected:
                print("WARNING: FFT produced non-exact result for this input size")
        else:
            t_fft = None
            print("FFT: not available (phase4.fft_multiply not importable)")

        # Karatsuba (skip big)
        if digits > skip_karatsuba_digits:
            t_kar = None
            print(f"Karatsuba: skipped for digits>{skip_karatsuba_digits}")
        else:
            t_kar, kar_t = time_fn(karatsuba_multiply, a, b, trials=trials)
            print(f"Karatsuba median: {t_kar:.6f}s  trials: {kar_t}")

        # Schoolbook
        t_sch, sch_t = time_fn(schoolbook_multiply, a, b, trials=trials)
        print(f"Schoolbook median: {t_sch:.6f}s  trials: {sch_t}")

        rows.append({
            "digits": digits,
            "base": base,
            "ntt_median_s": t_ntt,
            "fft_median_s": t_fft,
            "karatsuba_median_s": t_kar,
            "schoolbook_median_s": t_sch,
        })
    return rows


def _maybe_plot(rows, out_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot.")
        return
    digits = [r['digits'] for r in rows]
    ntt_times = [r['ntt_median_s'] for r in rows]
    fft_times = [r['fft_median_s'] if r['fft_median_s'] is not None else float('nan') for r in rows]
    kar_times = [r['karatsuba_median_s'] if r['karatsuba_median_s'] is not None else float('nan') for r in rows]
    sch_times = [r['schoolbook_median_s'] for r in rows]

    plt.figure()
    plt.loglog(digits, ntt_times, marker='o', label='NTT')
    if any(not math.isnan(x) for x in fft_times):
        plt.loglog(digits, fft_times, marker='o', label='FFT')
    if any(not math.isnan(x) for x in kar_times):
        plt.loglog(digits, kar_times, marker='o', label='Karatsuba')
    plt.loglog(digits, sch_times, marker='o', label='Schoolbook')
    plt.xlabel('Decimal digits')
    plt.ylabel('Median time (s)')
    plt.title('Phase5: Multiplication benchmarks (NTT vs FFT vs others)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NTT vs FFT vs Schoolbook")
    parser.add_argument("--sizes", nargs="*", type=int, default=[10, 50, 200, 800],
                        help="list of decimal-digit sizes to test")
    parser.add_argument("--trials", type=int, default=3, help="trials per test")
    parser.add_argument("--out", type=str, default="phase5/bench_ntt_results.csv", help="CSV output")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--base", type=int, default=1000, help="digit base for splitting")
    parser.add_argument("--skip-karatsuba-digits", type=int, default=5000, help="skip karatsuba above this size")
    parser.add_argument("--plot", action='store_true', help="generate PNG plot")
    parser.add_argument("--plot-out", type=str, default="phase5/bench_ntt_plot.png", help="plot output path")
    args = parser.parse_args()

    rows = bench_sizes(args.sizes, trials=args.trials, base_seed=args.seed, base=args.base, skip_karatsuba_digits=args.skip_karatsuba_digits)

    # write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["digits", "base", "ntt_median_s", "fft_median_s", "karatsuba_median_s", "schoolbook_median_s"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nResults written to {args.out}")

    if args.plot:
        _maybe_plot(rows, args.plot_out)

if __name__ == '__main__':
    main()

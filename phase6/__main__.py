# phase6/__main__.py
"""
Entry point for Phase 6: Auto-multiply benchmarking.
Running `python -m phase6` will execute benchmarks that compare:
 - auto_multiply
 - explicit FFT (if available)
 - explicit NTT (if available)
 - Karatsuba & Schoolbook

Outputs CSV + optional PNG plot.
"""

import argparse
import csv
import math
import random
import time
from typing import List

from .auto_multiply import auto_multiply, choose_algorithm
# try importing explicit implementations for comparison
try:
    from phase4.fft_multiply import multiply_ints as multiply_ints_fft
    _HAVE_FFT = True
except Exception:
    _HAVE_FFT = False

try:
    from phase5.ntt_multiply import multiply_ints_ntt
    _HAVE_NTT = True
except Exception:
    _HAVE_NTT = False

# local fallbacks
from .auto_multiply import karatsuba_multiply, schoolbook_multiply


def time_fn(fn, *args, trials=3):
    # simple timing utility
    # warmup
    fn(*args)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times)//2]


def bench_sizes(sizes: List[int], trials: int = 3, exact=False, prefer='auto'):
    rows = []
    for digits in sizes:
        a = random.randint(10**(digits-1), 10**digits - 1)
        b = random.randint(10**(digits-1), 10**digits - 1)
        expected = a * b
        print(f"\n=== digits={digits} ===")

        # auto
        t_auto = time_fn(auto_multiply, a, b, {'exact': exact, 'prefer': prefer}, trials=trials) if False else None
        # The auto_multiply wrapper needs keyword args; measure explicitly via lambda
        t_auto = time_fn(lambda x=a, y=b: auto_multiply(x, y, exact=exact, prefer=prefer), trials=trials)
        r_auto = auto_multiply(a, b, exact=exact, prefer=prefer)
        print(f"auto_multiply median: {t_auto:.6f}s (alg={choose_algorithm(a,b,exact=exact,prefer=prefer)})")

        # FFT
        if _HAVE_FFT:
            t_fft = time_fn(multiply_ints_fft, a, b, trials=trials)
            r_fft = multiply_ints_fft(a, b)
            print(f"FFT median: {t_fft:.6f}s")
        else:
            t_fft = None
            print("FFT: not available")

        # NTT
        if _HAVE_NTT:
            t_ntt = time_fn(multiply_ints_ntt, a, b, trials=trials)
            r_ntt = multiply_ints_ntt(a, b)
            print(f"NTT median: {t_ntt:.6f}s")
        else:
            t_ntt = None
            print("NTT: not available")

        # Karatsuba
        t_kar = time_fn(karatsuba_multiply, a, b, trials=trials)
        print(f"Karatsuba median: {t_kar:.6f}s")

        # Schoolbook
        t_sch = time_fn(schoolbook_multiply, a, b, trials=trials)
        print(f"Schoolbook median: {t_sch:.6f}s")

        rows.append({
            'digits': digits,
            'auto_s': t_auto,
            'fft_s': t_fft,
            'ntt_s': t_ntt,
            'karatsuba_s': t_kar,
            'schoolbook_s': t_sch,
            'chosen_by_auto': choose_algorithm(a,b,exact=exact,prefer=prefer)
        })
    return rows


def maybe_plot(rows, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available; skip plot')
        return
    digits = [r['digits'] for r in rows]
    plt.figure()
    plt.loglog(digits, [r['auto_s'] for r in rows], marker='o', label='auto')
    if any(r['fft_s'] is not None for r in rows):
        plt.loglog(digits, [r['fft_s'] or float('nan') for r in rows], marker='o', label='fft')
    if any(r['ntt_s'] is not None for r in rows):
        plt.loglog(digits, [r['ntt_s'] or float('nan') for r in rows], marker='o', label='ntt')
    plt.loglog(digits, [r['karatsuba_s'] for r in rows], marker='o', label='karatsuba')
    plt.loglog(digits, [r['schoolbook_s'] for r in rows], marker='o', label='schoolbook')
    plt.xlabel('decimal digits')
    plt.ylabel('median time (s)')
    plt.title('Phase6: auto_multiply benchmark')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Plot saved to {out_png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='*', type=int, default=[10,50,200,800,2000,8000], help='decimal digit sizes')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--out', type=str, default='phase6/bench_auto_results.csv')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-out', type=str, default='phase6/bench_auto_plot.png')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--prefer', type=str, default='auto')
    args = parser.parse_args()

    rows = bench_sizes(args.sizes, trials=args.trials, exact=args.exact, prefer=args.prefer)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['digits','auto_s','fft_s','ntt_s','karatsuba_s','schoolbook_s','chosen_by_auto'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'\nResults written to {args.out}')
    if args.plot:
        maybe_plot(rows, args.plot_out)

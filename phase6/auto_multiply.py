# phase6/auto_multiply.py
"""
Hybrid auto-selector for big-integer multiplication (Phase 6).

Provides a single API `auto_multiply(x, y, *, exact=False, prefer='auto', base=1000, debug=False)`
that chooses among:
 - schoolbook (very small sizes)
 - karatsuba (small/medium)
 - FFT (fast approximate using phase4.fft_multiply)
 - NTT (exact, using phase5.ntt_multiply)

Selection rules (configurable by thresholds below):
 - If exact=True -> prefer NTT (exact) for large sizes, or schoolbook/karatsuba for tiny ones.
 - If prefer='fft'/'ntt'/'karatsuba'/'schoolbook' -> force choice when feasible.
 - Otherwise, automatic heuristic based on decimal digit length.

This file also includes a basic CLI to run quick sanity checks and micro-benchmarks.

"""

import math
import time
from typing import Tuple
import sys
try:
    sys.set_int_max_str_digits(1000000)
except AttributeError:
    pass  # older Python versions don’t need this

# Try importing available implementations
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

# Local fallbacks (schoolbook and karatsuba implementations)
# Borrowed/compatible with earlier phase code

def _to_digits(n: int, base: int):
    if n == 0:
        return [0]
    digs = []
    while n:
        digs.append(n % base)
        n //= base
    return digs


def schoolbook_multiply(a: int, b: int, base: int = 10000) -> int:
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


# ---------------------
# Heuristics / thresholds
# ---------------------
# Default thresholds (will be overwritten if phase6/thresholds.json exists)
_SMALL_DIGITS = 100         # up to 100 digits -> schoolbook
_MEDIUM_DIGITS = 5000       # up to 5k -> karatsuba
_FFT_PREFERRED = 5000       # above this FFT usually wins in speed
_NTT_PREFERRED = 100000     # above this consider NTT for exactness

# Attempt to load empirically calibrated thresholds from phase6/thresholds.json
# Format expected: {"SMALL_DIGITS": int, "MEDIUM_DIGITS": int, "FFT_PREFERRED": int, "NTT_PREFERRED": int}
try:
    import json
    from pathlib import Path

    _THRESH_PATH = Path(__file__).parent / "thresholds.json"
    if _THRESH_PATH.exists():
        with _THRESH_PATH.open("r", encoding="utf-8") as _f:
            _data = json.load(_f)
        # replace defaults, but be defensive about types
        if isinstance(_data, dict):
            _SMALL_DIGITS = int(_data.get("SMALL_DIGITS", _SMALL_DIGITS))
            _MEDIUM_DIGITS = int(_data.get("MEDIUM_DIGITS", _MEDIUM_DIGITS))
            _FFT_PREFERRED = int(_data.get("FFT_PREFERRED", _FFT_PREFERRED))
            _NTT_PREFERRED = int(_data.get("NTT_PREFERRED", _NTT_PREFERRED))
except Exception:
    # If anything goes wrong reading/parsing the file, keep defaults silently.
    pass


# ---------------------
# Auto-select logic
# ---------------------

def _digits_count(x: int) -> int:
    if x == 0:
        return 1
    return int(x.bit_length() / math.log2(10)) + 1


def choose_algorithm(x: int, y: int, exact: bool = False, prefer: str = 'auto') -> str:
    """Return one of: 'schoolbook', 'karatsuba', 'fft', 'ntt'"""
    dx = _digits_count(abs(x))
    dy = _digits_count(abs(y))
    digits = max(dx, dy)

    if prefer and prefer != 'auto':
        pref = prefer.lower()
        if pref == 'ntt' and _HAVE_NTT:
            return 'ntt'
        if pref == 'fft' and _HAVE_FFT:
            return 'fft'
        if pref == 'karatsuba':
            return 'karatsuba'
        if pref == 'schoolbook':
            return 'schoolbook'
        # fallback to auto if preferred algorithm isn't available

    if exact:
        # prefer exact algorithms for large sizes
        if digits <= _SMALL_DIGITS:
            return 'schoolbook'
        if digits <= _MEDIUM_DIGITS:
            return 'karatsuba'
        # prefer NTT if available else FFT
        return 'ntt' if _HAVE_NTT else ('fft' if _HAVE_FFT else 'karatsuba')

    # not exact: pick by speed heuristics
    if digits <= _SMALL_DIGITS:
        return 'schoolbook'
    if digits <= _MEDIUM_DIGITS:
        return 'karatsuba'
    # large sizes: FFT if available (fast), else NTT if available, else karatsuba
    if _HAVE_FFT:
        return 'fft'
    if _HAVE_NTT:
        return 'ntt'
    return 'karatsuba'


def auto_multiply(x: int, y: int, *, exact: bool = False, prefer: str = 'auto', base: int = 1000, debug: bool = False) -> int:
    """Multiply two integers using auto-selected algorithm.

    Parameters:
      exact: if True, prefer exact algorithms (NTT) for large sizes.
      prefer: 'auto' or one of 'fft','ntt','karatsuba','schoolbook'. Forces choice if available.
      base: base parameter passed to some algorithms if used (currently unused except in schoolbook wrapper).
      debug: print selection information.
    """
    alg = choose_algorithm(x, y, exact=exact, prefer=prefer)
    if debug:
        print(f"auto_multiply: chosen algorithm={alg} (exact={exact}, prefer={prefer})")

    if alg == 'schoolbook':
        return schoolbook_multiply(x, y, base=base)
    if alg == 'karatsuba':
        return karatsuba_multiply(x, y)
    if alg == 'fft':
        if not _HAVE_FFT:
            # fallback
            return karatsuba_multiply(x, y)
        return multiply_ints_fft(x, y)
    if alg == 'ntt':
        if not _HAVE_NTT:
            # fallback
            if _HAVE_FFT:
                return multiply_ints_fft(x, y)
            return karatsuba_multiply(x, y)
        return multiply_ints_ntt(x, y, base=base)

    # Fallback
    return x * y

# ---------------------
# CLI / quick bench
# ---------------------
if __name__ == "__main__":
    import math
    import time

    tests = [10, 50, 200, 800, 2000, 8000]
    # Add an extra very large stress test (50k digits). Keep as a single trial.
    heavy_tests = [50000]

    print("Phase 6: Auto-multiply quick benchmark")
    for d in tests:
        a = int("9" * d)
        b = int("8" * d)
        t0 = time.perf_counter()
        r = auto_multiply(a, b)
        t1 = time.perf_counter()

        # safer digit-length check without str()
        if r == 0:
            result_len = 1
        else:
            result_len = int(math.log10(abs(r))) + 1
        expected_len_range = (d * 2 - 1, d * 2)

        print(
            f"digits={d} -> time {t1 - t0:.6f}s, "
            f"product_has_correct_length={result_len in expected_len_range}"
        )

    # Heavy single-shot test (very large) — optional, may take time & memory.
    for d in heavy_tests:
        print("\n*** HEAVY TEST: digits={} — single trial. Hit Ctrl-C to abort if too slow ***".format(d))
        a = int("9" * d)
        b = int("8" * d)
        t0 = time.perf_counter()
        r = auto_multiply(a, b)
        t1 = time.perf_counter()
        if r == 0:
            result_len = 1
        else:
            result_len = int(math.log10(abs(r))) + 1
        expected_len_range = (d * 2 - 1, d * 2)
        print(f"HEAVY digits={d} -> time {t1 - t0:.3f}s, product_has_correct_length={result_len in expected_len_range}")


    import argparse
    import random
    parser = argparse.ArgumentParser(description='Phase6: Hybrid auto-multiply')
    parser.add_argument('--sizes', nargs='*', type=int, default=[10, 50, 200, 800, 2000, 8000], help='decimal digit sizes to test')
    parser.add_argument('--trials', type=int, default=3, help='trials per size')
    parser.add_argument('--exact', action='store_true', help='prefer exact algorithms (NTT)')
    parser.add_argument('--prefer', type=str, default='auto', help="force prefer: 'auto','fft','ntt','karatsuba','schoolbook'")
    args = parser.parse_args()

    print('Phase 6: Auto-multiply quick benchmark')
    for d in args.sizes:
        a = random.randint(10**(d-1), 10**d - 1)
        b = random.randint(10**(d-1), 10**d - 1)
        # time auto_multiply
        t0 = time.perf_counter()
        r = auto_multiply(a, b, exact=args.exact, prefer=args.prefer, debug=True)
        t1 = time.perf_counter()
        print(f'digits={d} -> time {t1-t0:.6f}s, product_has_correct_length={len(str(abs(r))) in (d*2-1, d*2)}')

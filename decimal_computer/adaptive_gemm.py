"""
Adaptive GEMM / Big Integer Multiplication Selector
---------------------------------------------------
This layer automatically chooses the best multiplication backend:
- Schoolbook for small sizes
- Karatsuba (nogil) for medium sizes
- (Future) FFT/NTT for very large operands

Author: Karan Singh & Aladdin Engine
"""

import math
import time
from typing import Callable

# Optional: for small fallback multiply (Python-based)
def schoolbook_mul(a: int, b: int) -> int:
    return a * b

# Try to import karatsuba backend (graceful fallback)
try:
    from phase8_karatsuba_nogil import multiply as karatsuba_mul  # type: ignore
    HAVE_KARATSUBA = True
except Exception:
    karatsuba_mul = None  # type: ignore
    HAVE_KARATSUBA = False

# thresholds (bit lengths) — tune to your platform
SCHOOLBOOK_LIMIT = 4096       # below this, use Python * (fast enough)
KARATSUBA_LIMIT = 32768       # up to this, use Karatsuba
FFT_LIMIT = 262144            # beyond this (future) → FFT/NTT

def adaptive_mul(a: int, b: int) -> int:
    """Choose best multiplication algorithm adaptively."""
    # handle trivial zeros quickly
    if a == 0 or b == 0:
        return 0

    nbits = max(a.bit_length(), b.bit_length())

    # Very small: Python builtin (often fastest due to C bignum implementation)
    if nbits <= SCHOOLBOOK_LIMIT:
        backend = "schoolbook (python *)"
        result = schoolbook_mul(a, b)

    # Karatsuba region
    elif nbits <= KARATSUBA_LIMIT:
        if HAVE_KARATSUBA:
            backend = "karatsuba (nogil)"
            result = karatsuba_mul(a, b)
        else:
            backend = "fallback schoolbook (karatsuba missing)"
            result = schoolbook_mul(a, b)

    # Large inputs: for now use Karatsuba if available, else fallback
    else:
        if HAVE_KARATSUBA:
            backend = "karatsuba (nogil) — large"
            result = karatsuba_mul(a, b)
        else:
            backend = "fallback schoolbook (karatsuba missing) — large"
            result = schoolbook_mul(a, b)

    # optional: lightweight sanity check during development (can be removed)
    # if result != a * b:
    #     raise RuntimeError(f"Multiplication mismatch using {backend}")

    # return result; caller may log backend if desired
    return result

def adaptive_mul_with_log(a: int, b: int) -> (int, str):
    """Like adaptive_mul but also returns the backend string for logging."""
    if a == 0 or b == 0:
        return 0, "zero-shortcut"
    nbits = max(a.bit_length(), b.bit_length())
    if nbits <= SCHOOLBOOK_LIMIT:
        return schoolbook_mul(a, b), "schoolbook (python *)"
    elif nbits <= KARATSUBA_LIMIT:
        if HAVE_KARATSUBA:
            return karatsuba_mul(a, b), "karatsuba (nogil)"
        else:
            return schoolbook_mul(a, b), "fallback schoolbook"
    else:
        if HAVE_KARATSUBA:
            return karatsuba_mul(a, b), "karatsuba (nogil) — large"
        else:
            return schoolbook_mul(a, b), "fallback schoolbook — large"


# -----------------------------------------------------------------
# Simple benchmark / self-test
# -----------------------------------------------------------------
if __name__ == "__main__":
    sizes = [512, 4096, 16384, 65536]
    print("Karatsuba available:", HAVE_KARATSUBA)
    for bits in sizes:
        a = (1 << bits) - 12345
        b = (1 << bits) - 6789
        t0 = time.perf_counter()
        res, backend = adaptive_mul_with_log(a, b)
        t1 = time.perf_counter()
        ok = (res == a * b)
        print(f"{bits:6d} bits: time={t1-t0:.6f}s  OK={ok}  backend={backend}")

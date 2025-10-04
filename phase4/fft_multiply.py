"""
fft_multiply.py

Provide FFT-based convolution/multiplication utilities.

Behavior:
- If numpy is installed, uses numpy.fft.fft/ifft for speed.
- Otherwise falls back to a pure-Python Cooley-Tukey FFT implementation (radix-2 iterative).
- Exports:
    fft_convolve(a, b) -> list of floats (length = len(a)+len(b)-1)
    multiply_polynomials(a, b) -> list of ints (rounded coefficients)
    multiply_ints(x, y) -> int (product)

Notes on precision:
- For integer multiplication we split the integers into base BASE digits (BASE=1000 by default)
  to reduce numeric errors and then handle carries carefully.
- This implementation favors correctness and compatibility over extreme micro-optimizations.
"""

from math import ceil, log2

try:
    import numpy as _np
    _USE_NUMPY = True
except Exception:
    _USE_NUMPY = False

import cmath
import math
from typing import List, Sequence, Union

Number = Union[int, float, complex]

# ----------------------
# Utilities
# ----------------------

def _next_power_of_two(n: int) -> int:
    return 1 << (n-1).bit_length()

# ----------------------
# Pure-Python FFT fallback (iterative Cooley-Tukey, radix-2)
# ----------------------

def _py_fft(a: List[complex], invert: bool=False) -> List[complex]:
    """
    In-place iterative Cooley-Tukey radix-2 FFT (returns new list)
    invert=False -> FFT
    invert=True  -> IFFT (needs division by n by caller)
    Note: expects len(a) to be a power of two.
    """
    n = len(a)
    A = list(a)  # copy
    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            A[i], A[j] = A[j], A[i]

    length = 2
    while length <= n:
        ang = 2 * math.pi / length * (-1 if not invert else 1)
        wlen = complex(math.cos(ang), math.sin(ang))
        for i in range(0, n, length):
            w = 1+0j
            half = length >> 1
            for j in range(i, i + half):
                u = A[j]
                v = A[j + half] * w
                A[j] = u + v
                A[j + half] = u - v
                w *= wlen
        length <<= 1

    if invert:
        # caller may divide; but let's divide here for convenience
        A = [x / n for x in A]
    return A

# ----------------------
# FFT wrapper (numpy or fallback)
# ----------------------

def _fft(a: Sequence[complex]) -> List[complex]:
    if _USE_NUMPY:
        return list(_np.fft.fft(_np.asarray(a, dtype=_np.complex128)))
    else:
        return _py_fft(list(a), invert=False)

def _ifft(A: Sequence[complex]) -> List[complex]:
    if _USE_NUMPY:
        return list(_np.fft.ifft(_np.asarray(A, dtype=_np.complex128)))
    else:
        return _py_fft(list(A), invert=True)

# ----------------------
# Convolution
# ----------------------

def fft_convolve(a: Sequence[Number], b: Sequence[Number]) -> List[float]:
    """
    Convolve two real sequences using FFT.
    Returns list of floats of length len(a) + len(b) - 1.
    """
    na = len(a)
    nb = len(b)
    if na == 0 or nb == 0:
        return []

    n = _next_power_of_two(na + nb - 1)
    # prepare complex arrays
    A = [complex(x) for x in a] + [0] * (n - na)
    B = [complex(x) for x in b] + [0] * (n - nb)

    FA = _fft(A)
    FB = _fft(B)

    FC = [FA[i] * FB[i] for i in range(n)]
    c = _ifft(FC)
    # take only real part and required length
    result = [c[i].real for i in range(na + nb - 1)]
    return result

# ----------------------
# Polynomial multiplication (integer coefficients)
# ----------------------

def multiply_polynomials(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """
    Multiply two polynomials with integer coefficients given as lists (lowest-first).
    Returns integer coefficient list (rounded).
    Example: multiply_polynomials([1,2], [3,4]) -> [3,10,8]  (since (1+2x)*(3+4x)=3+10x+8x^2)
    """
    if not a or not b:
        return []
    conv = fft_convolve(a, b)
    # round to nearest int
    return [int(round(x)) for x in conv]

# ----------------------
# Integer multiplication via convolution
# ----------------------

def multiply_ints(x: int, y: int, base: int = 1000) -> int:
    """
    Multiply two integers using FFT-based convolution.
    - base: digit base to split integers into; base=1000 (3 decimal digits) is conservative.
    Returns the exact integer product.
    """
    if x == 0 or y == 0:
        return 0

    sign = -1 if (x < 0) ^ (y < 0) else 1
    ax = abs(x)
    ay = abs(y)

    # Convert to digit arrays in given base (least-significant digit first)
    def to_digits(n):
        if n == 0:
            return [0]
        digs = []
        while n:
            digs.append(n % base)
            n //= base
        return digs

    da = to_digits(ax)
    db = to_digits(ay)

    # Convolve
    conv = fft_convolve(da, db)
    # Round to nearest integer
    conv_int = [int(round(v)) for v in conv]

    # Handle carries
    carry = 0
    for i in range(len(conv_int)):
        total = conv_int[i] + carry
        carry = total // base
        conv_int[i] = total % base
    while carry:
        conv_int.append(carry % base)
        carry //= base

    # Convert back to integer
    result = 0
    for d in reversed(conv_int):
        result = result * base + d

    return sign * result

# ----------------------
# Small test harness
# ----------------------

def _self_test():
    import random, sys
    print("numpy available:", _USE_NUMPY)
    # small polynomial test
    p1 = [1, 2, 3]   # 1 + 2x + 3x^2
    p2 = [4, 5]      # 4 + 5x
    print("polys", p1, "*", p2, "=", multiply_polynomials(p1, p2))  # expect [4,13,22,15]

    # small integer tests
    tests = [
        (0, 12345),
        (123, 456),
        (99999, 0),
        (123456789, 987654321),
    ]
    for a, b in tests:
        print(f"{a} * {b} = {multiply_ints(a,b)} (expected {a*b})")

    # random stress test (small)
    for _ in range(10):
        a = random.randint(-10**12, 10**12)
        b = random.randint(-10**12, 10**12)
        prod = multiply_ints(a, b)
        if prod != a * b:
            print("ERROR:", a, b, prod, a*b)
            sys.exit(1)
    print("Random integer tests passed.")

    # larger integer test (bigger than float 53-bit)
    a = 2**80 + 12345678901234567890
    b = 2**70 + 9876543210987654321
    got = multiply_ints(a, b)
    expect = a * b
    print("Large multiply pass?", got == expect)

if __name__ == "__main__":
    _self_test()

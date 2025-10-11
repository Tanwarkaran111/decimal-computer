# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
phase4/fft_multiply_fast.pyx

Safe, compilable NumPy + Cython hybrid FFT integer multiplication.
This version avoids fragile C-API signatures and compiles reliably.
"""

# Python imports
import numpy as np
import math
cimport cython

# -------------------------
# Helpers (pure-Python / NumPy friendly)
# -------------------------
def _to_digits_py(n, base):
    """Convert Python int -> Python list of digits (LSB-first)."""
    if n == 0:
        return [0]
    digs = []
    x = n
    b = base
    while x:
        digs.append(int(x % b))
        x //= b
    return digs

def _from_digits_py(digs, base):
    """Convert an iterable of digits (LSB-first) back to a Python int."""
    res = 0
    b = base
    for d in reversed(digs):
        res = res * b + int(d)
    return res

def _normalize_py(arr, base):
    """
    Normalize 1D numpy array or sequence of ints so each element < base.
    Returns a numpy.int64 array of normalized limbs (LSB-first).
    Implemented with Python-int carries for correctness.
    """
    a = np.asarray(arr, dtype=np.int64)
    py = [int(a[i]) for i in range(a.shape[0])]
    carry = 0
    B = int(base)
    for i in range(len(py)):
        total = py[i] + carry
        carry = total // B
        py[i] = total - carry * B
    while carry:
        py.append(carry % B)
        carry //= B
    return np.array(py, dtype=np.int64)

# -------------------------
# Public API
# -------------------------
def multiply_ints(x, y, base: int = 1000):
    """
    Multiply two Python integers using NumPy FFT convolution and carry normalization.
    - base: digit base to split integers (default 1000).
    Returns exact Python int.
    """
    if x == 0 or y == 0:
        return 0

    sign = 1
    if x < 0:
        x = -x
        sign = -sign
    if y < 0:
        y = -y
        sign = -sign

    # convert to digits (LSB-first)
    a_list = _to_digits_py(x, base)
    b_list = _to_digits_py(y, base)

    # convert to numpy float arrays for FFT
    a_arr = np.asarray(a_list, dtype=np.float64)
    b_arr = np.asarray(b_list, dtype=np.float64)

    needed = a_arr.size + b_arr.size - 1
    # next power of two
    n = 1 << ((needed - 1).bit_length())

    # perform convolution using rfft/irfft (real-fft)
    fa = np.fft.rfft(a_arr, n)
    fb = np.fft.rfft(b_arr, n)
    fc = fa * fb
    conv = np.fft.irfft(fc, n)

    # take only needed length, round to nearest integer, cast to int64
    conv_rounded = np.rint(conv[:needed]).astype(np.int64)

    # normalize carries to ensure each limb < base
    normalized = _normalize_py(conv_rounded, base)

    # convert back to Python int
    result = _from_digits_py(normalized, base)
    return result if sign > 0 else -result

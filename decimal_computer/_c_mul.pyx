# src/decimal_computer/_c_mul.pyx
# cython: language_level=3
"""
Tiny Cython helper: compute elementwise integer products in C and return a Python list[int].

This avoids Python-level multiply/loop overhead in pure-Python hot loops.
We intentionally do NOT implement quantize here to keep rounding semantics centralized
in Python (quantize_int). This keeps changes small and safe.
"""
from cpython.ref cimport PyObject
from libcpp.vector cimport vector
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def mul_products(list a_vals, list b_vals):
    """
    Multiply two Python lists of integers elementwise and return a Python list of ints.
    - Expects len(a_vals) == len(b_vals).
    - Uses C-level loop to speed repeated multiplication.
    """
    cdef Py_ssize_t n = len(a_vals)
    if n != len(b_vals):
        raise ValueError("a_vals and b_vals must have the same length")

    # allocate result list with known size
    res = [0] * n
    cdef Py_ssize_t i
    for i in range(n):
        # use Python int multiply but inside C loop (avoids Python-level loop overhead)
        res[i] = a_vals[i] * b_vals[i]
    return res

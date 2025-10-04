# tests/test_numba_decimal.py
from numba_decimal import int_to_digits_numba, digits_to_int_numba, naive_digit_mul_numba
import numpy as np

def test_digits_roundtrip():
    x = 12345
    d = int_to_digits_numba(x, 8)
    assert digits_to_int_numba(d) == x

def test_mul():
    a = 213
    b = 42
    ad = int_to_digits_numba(a, 8)
    bd = int_to_digits_numba(b, 8)
    c_digits = naive_digit_mul_numba(ad, bd)
    assert digits_to_int_numba(c_digits) == a * b

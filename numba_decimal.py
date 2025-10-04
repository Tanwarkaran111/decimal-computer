# numba_decimal.py
import numpy as np
from numba import njit, prange, int64

BASE = 10  # decimal base

@njit
def int_to_digits_numba(x, ndigits):
    # produce array of length ndigits (LSB first)
    out = np.zeros(ndigits, np.int64)
    tmp = x
    for i in range(ndigits):
        out[i] = tmp % BASE
        tmp //= BASE
    return out

@njit
def digits_to_int_numba(digits):
    res = 0
    powb = 1
    for i in range(len(digits)):
        res += int(digits[i]) * powb
        powb *= BASE
    return res

@njit
def digit_add_numba(A, B):
    # A,B are digit arrays (LSB-first), returns (C,c)
    n = max(len(A), len(B))
    C = np.zeros(n+1, np.int64)
    carry = 0
    for i in range(n):
        ai = A[i] if i < len(A) else 0
        bi = B[i] if i < len(B) else 0
        s = ai + bi + carry
        C[i] = s % BASE
        carry = s // BASE
    C[n] = carry
    # trim?
    return C

@njit
def digit_mul_word_numba(A, b):
    # multiply digit-array A by small digit b (0..BASE-1)
    n = len(A)
    C = np.zeros(n+1, np.int64)
    carry = 0
    for i in range(n):
        t = A[i] * b + carry
        C[i] = t % BASE
        carry = t // BASE
    C[n] = carry
    return C

@njit
def naive_digit_mul_numba(A, B):
    # full schoolbook multiplication of two digit-arrays
    na = len(A)
    nb = len(B)
    C = np.zeros(na+nb, np.int64)
    for i in range(na):
        carry = 0
        for j in range(nb):
            t = C[i+j] + A[i] * B[j] + carry
            C[i+j] = t % BASE
            carry = t // BASE
        C[i+nb] += carry
    return C

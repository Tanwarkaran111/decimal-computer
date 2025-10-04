# decimal_digit_starter.py
# Digit-level helpers with counters + Karatsuba digit multiply.

from typing import List, Tuple
from math import ceil

# Global counters
DIGIT_MUL_COUNT = 0
DIGIT_ADD_COUNT = 0

def reset_counters() -> None:
    global DIGIT_MUL_COUNT, DIGIT_ADD_COUNT
    DIGIT_MUL_COUNT = 0
    DIGIT_ADD_COUNT = 0

def get_counters() -> Tuple[int, int]:
    return DIGIT_MUL_COUNT, DIGIT_ADD_COUNT

def _inc_mul(n: int = 1) -> None:
    global DIGIT_MUL_COUNT
    DIGIT_MUL_COUNT += n

def _inc_add(n: int = 1) -> None:
    global DIGIT_ADD_COUNT
    DIGIT_ADD_COUNT += n

def int_to_digits(n: int) -> List[int]:
    assert n >= 0
    if n == 0:
        return [0]
    digits: List[int] = []
    while n:
        digits.append(n % 10)
        n //= 10
    return digits

def digits_to_int(digits: List[int]) -> int:
    out = 0
    mul = 1
    for d in digits:
        out += d * mul
        mul *= 10
    return out

def digit_add(A: List[int], B: List[int]) -> Tuple[List[int], int]:
    n = max(len(A), len(B))
    S: List[int] = []
    carry = 0
    for i in range(n):
        a = A[i] if i < len(A) else 0
        b = B[i] if i < len(B) else 0
        _inc_add(1)
        s = a + b + carry
        S.append(s % 10)
        carry = s // 10
    return S, carry

def _add_full(A: List[int], B: List[int]) -> List[int]:
    """Return full sum digit-list (handles carry as extra digit). Uses digit_add (counts adds)."""
    S, carry = digit_add(A, B)
    if carry:
        S.extend(int_to_digits(carry))
    return S

def digit_mul(A: List[int], B: List[int]) -> List[int]:
    # schoolbook digit multiplication
    m = len(A)
    n = len(B)
    P = [0] * (m + n)
    for i in range(m):
        carry = 0
        for j in range(n):
            _inc_mul(1)         # single-digit multiply
            _inc_add(1)         # for accumulation P[i+j] + prod + carry
            total = P[i + j] + A[i] * B[j] + carry
            P[i + j] = total % 10
            carry = total // 10
        k = i + n
        while carry:
            _inc_add(1)
            total = P[k] + carry
            P[k] = total % 10
            carry = total // 10
            k += 1
    while len(P) > 1 and P[-1] == 0:
        P.pop()
    return P

# -------- Karatsuba for digit lists (lsf = least-significant-first) ----------
def _strip_leading_zeros(P: List[int]) -> List[int]:
    while len(P) > 1 and P[-1] == 0:
        P.pop()
    return P

def _shift_digits(A: List[int], k: int) -> List[int]:
    """Multiply A by 10**k (shift least-significant-first)"""
    if not any(A):
        return [0]
    return [0]*k + A

def digit_mul_karatsuba(A: List[int], B: List[int], cutoff: int = 8) -> List[int]:
    """
    Karatsuba multiplication on digit-lists (least-significant-first).
    `cutoff` = when max(len(A),len(B)) <= cutoff, use schoolbook to avoid recursion overhead.
    Counts elementary digit mul/add operations via the same counters (we use digit_mul and digit_add).
    """
    m = len(A)
    n = len(B)
    # base cases
    if m == 0 or n == 0:
        return [0]
    if max(m, n) <= cutoff:
        return digit_mul(A, B)  # schoolbook path (already counts ops)

    # Make lengths equal by padding
    N = max(m, n)
    half = N // 2

    # split A into A_low (digits 0..half-1) and A_high (half..end)
    A_low = A[:half] if half <= len(A) else A[:]  # may be shorter
    A_high = A[half:] if half < len(A) else [0]

    B_low = B[:half] if half <= len(B) else B[:]
    B_high = B[half:] if half < len(B) else [0]

    # recursively compute z0 = low*low, z2 = high*high
    z0 = digit_mul_karatsuba(A_low, B_low, cutoff=cutoff)
    z2 = digit_mul_karatsuba(A_high, B_high, cutoff=cutoff)

    # (A_low + A_high) and (B_low + B_high)
    sum_A = _add_full(A_low, A_high)
    sum_B = _add_full(B_low, B_high)

    z1 = digit_mul_karatsuba(sum_A, sum_B, cutoff=cutoff)

    # z1 - z2 - z0  (subtractions done via integer conversion for simplicity),
    # but we need to keep digit counters only for additions/mults; subtractions do not change our counters
    # We'll convert to ints, subtract, then back to digits (this does not increment our counters).
    z0_int = digits_to_int(z0)
    z1_int = digits_to_int(z1)
    z2_int = digits_to_int(z2)
    middle_int = z1_int - z2_int - z0_int
    # combine: result = z2 * 10^(2*half) + middle * 10^half + z0
    res_int = z2_int * (10 ** (2*half)) + middle_int * (10 ** half) + z0_int
    res_digits = int_to_digits(res_int)
    return _strip_leading_zeros(res_digits)

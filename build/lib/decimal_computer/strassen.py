# decimal_computer/strassen.py
"""
A small, robust Strassen matrix multiply wrapper for the decimal_computer prototype.

Usage:
    from decimal_computer.strassen import strassen_gemm
    C, counters = strassen_gemm(A, B, return_counters=True, cutoff=16)
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Tuple, Optional

# project helpers (relative import so `python -m decimal_computer.cli` works)
from .decimal_digit_starter import digits_to_int, int_to_digits
from .decimal_gemm import decimal_gemm_naive


Matrix = List[List[Any]]


def _is_digit_matrix(M: Matrix) -> bool:
    """Return True if a matrix cell looks like a digit-list (list/tuple of small ints)."""
    if not M or not M[0]:
        return False
    cell = M[0][0]
    return isinstance(cell, (list, tuple))


def _to_int_matrix(M: Matrix) -> List[List[int]]:
    """Convert matrix of digit-lists to matrix of ints; if already ints return as-is."""
    if _is_digit_matrix(M):
        return [[digits_to_int(cell) for cell in row] for row in M]
    return [[int(cell) for cell in row] for row in M]


def _to_digit_matrix(I: List[List[int]]) -> Matrix:
    """Convert integer matrix back to digit-lists using int_to_digits (preserves shape)."""
    return [[int_to_digits(cell) for cell in row] for row in I]


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pad_matrix(A: List[List[int]], size: int) -> List[List[int]]:
    n = len(A)
    m = len(A[0]) if n > 0 else 0
    B = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            B[i][j] = A[i][j]
    return B


def _unpad_matrix(A: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    return [row[:cols] for row in A[:rows]]


def _add_matrix(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _sub_matrix(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def _zero_matrix(n: int) -> List[List[int]]:
    return [[0] * n for _ in range(n)]


def _strassen_recursive(
    A: List[List[int]],
    B: List[List[int]],
    cutoff: int,
) -> Tuple[List[List[int]], int, int]:
    """
    Perform Strassen multiplication on square integer matrices A,B of size n=power-of-two.
    Returns (C, muls_count, adds_count) where counts are integer operation estimates
    (elementwise scalar muls/adds).
    NOTE: we count additions/subtractions performed explicitly in the algorithm; recursive
    base calls will be delegated to decimal_gemm_naive to obtain accurate counters.
    """
    n = len(A)
    # base case: delegate to decimal_gemm_naive if n <= cutoff
    if n <= cutoff:
        # decimal_gemm_naive is used here; it may accept return_counters or not.
        try:
            C_full = decimal_gemm_naive(A, B, reset_counters_before=True, return_counters=True)
            # decimal_gemm_naive may return (C, counters_dict) or (C,muls,adds,time)
            if isinstance(C_full, tuple) or isinstance(C_full, list):
                # normalize
                # cases: (C, counters_dict) or (C, muls, adds, time) or (C, muls, adds)
                if len(C_full) >= 2 and isinstance(C_full[1], dict):
                    C = C_full[0]
                    counters = C_full[1]
                    muls = int(counters.get("muls", 0))
                    adds = int(counters.get("adds", 0))
                    return C, muls, adds
                else:
                    C = C_full[0]
                    muls = int(C_full[1]) if len(C_full) > 1 else 0
                    adds = int(C_full[2]) if len(C_full) > 2 else 0
                    return C, muls, adds
            # fallback
            return C_full, 0, 0
        except TypeError:
            # older signature: call without return_counters and estimate ops
            C = decimal_gemm_naive(A, B, reset_counters_before=True)
            # we don't have counters; fallback to rough estimate: n^3 scalar muls, ~n^3 adds
            est = n * n * n
            return C, est, est

    # recursive case: split into quadrants
    m = n // 2
    A11 = [[A[i][j] for j in range(0, m)] for i in range(0, m)]
    A12 = [[A[i][j] for j in range(m, n)] for i in range(0, m)]
    A21 = [[A[i][j] for j in range(0, m)] for i in range(m, n)]
    A22 = [[A[i][j] for j in range(m, n)] for i in range(m, n)]

    B11 = [[B[i][j] for j in range(0, m)] for i in range(0, m)]
    B12 = [[B[i][j] for j in range(m, n)] for i in range(0, m)]
    B21 = [[B[i][j] for j in range(0, m)] for i in range(m, n)]
    B22 = [[B[i][j] for j in range(m, n)] for i in range(m, n)]

    # compute S1..S10 (elementwise adds/subs)
    S1 = _sub_matrix(B12, B22)
    S2 = _add_matrix(A11, A12)
    S3 = _add_matrix(A21, A22)
    S4 = _sub_matrix(B21, B11)
    S5 = _add_matrix(A11, A22)
    S6 = _add_matrix(B11, B22)
    S7 = _sub_matrix(A12, A22)
    S8 = _add_matrix(B21, B22)
    S9 = _sub_matrix(A11, A21)
    S10 = _add_matrix(B11, B12)

    # count the adds performed in forming S matrices: each add/sub touches m*m entries
    adds_S = 10 * (m * m)  # 10 S matrices computed (adds or subs)

    # Seven recursive multiplications (M1..M7) with their op counters
    M1, m1_mul, m1_add = _strassen_recursive(A11, S1, cutoff)
    M2, m2_mul, m2_add = _strassen_recursive(S2, B22, cutoff)
    M3, m3_mul, m3_add = _strassen_recursive(S3, B11, cutoff)
    M4, m4_mul, m4_add = _strassen_recursive(A22, S4, cutoff)
    M5, m5_mul, m5_add = _strassen_recursive(S5, S6, cutoff)
    M6, m6_mul, m6_add = _strassen_recursive(S7, S8, cutoff)
    M7, m7_mul, m7_add = _strassen_recursive(S9, S10, cutoff)

    # combine results to get C11, C12, C21, C22:
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 + M1 - M3 - M7
    # Each combine is elementwise adds/subs; count them:
    combine_ops = 0

    # helpers to compute elementwise combinations and count ops
    def _elem_op_add_count(X, Y):
        nloc = len(X)
        return [[X[i][j] + Y[i][j] for j in range(nloc)] for i in range(nloc)], nloc * nloc

    def _elem_op_sub_count(X, Y):
        nloc = len(X)
        return [[X[i][j] - Y[i][j] for j in range(nloc)] for i in range(nloc)], nloc * nloc

    # compute C11
    T1, c = _elem_op_add_count(M5, M4)
    combine_ops += c
    T2, c = _elem_op_sub_count(T1, M2)
    combine_ops += c
    C11, c = _elem_op_add_count(T2, M6)
    combine_ops += c

    # C12
    C12, c = _elem_op_add_count(M1, M2)
    combine_ops += c

    # C21
    C21, c = _elem_op_add_count(M3, M4)
    combine_ops += c

    # C22
    T3, c = _elem_op_add_count(M5, M1)
    combine_ops += c
    T4, c = _elem_op_sub_count(T3, M3)
    combine_ops += c
    C22, c = _elem_op_sub_count(T4, M7)
    combine_ops += c

    # now assemble C from quadrants
    C = _zero_matrix(n)
    for i in range(m):
        for j in range(m):
            C[i][j] = C11[i][j]
            C[i][j + m] = C12[i][j]
            C[i + m][j] = C21[i][j]
            C[i + m][j + m] = C22[i][j]

    # sum the counts from recursive calls + S formation + combine_ops
    muls = m1_mul + m2_mul + m3_mul + m4_mul + m5_mul + m6_mul + m7_mul
    adds = m1_add + m2_add + m3_add + m4_add + m5_add + m6_add + m7_add
    adds += adds_S + combine_ops

    return C, muls, adds


def strassen_gemm(
    A_in: Matrix,
    B_in: Matrix,
    return_counters: bool = False,
    cutoff: Optional[int] = None,
):
    """
    Public wrapper for Strassen multiply.

    Parameters:
      A_in, B_in: input matrices; entries can be ints or digit-lists (digit-lists will be converted)
      return_counters: if True returns (C, counters_dict) where counters_dict contains
                       keys "muls","adds","time_s"
      cutoff: recursion cutoff (operate with decimal_gemm_naive when submatrix size <= cutoff).
              If None, use cutoff = 1 (i.e., recurse down to n==1).

    Returns:
      - If return_counters is False: C (digit-list matrix if inputs were digit-lists)
      - If True: (C, {"muls":int, "adds":int, "time_s":float})
    """
    start = perf_counter()

    # convert inputs to integer matrices (so Strassen arithmetic is on ints)
    intA = _to_int_matrix(A_in)
    intB = _to_int_matrix(B_in)

    nA_rows = len(intA)
    nA_cols = len(intA[0]) if nA_rows > 0 else 0
    nB_rows = len(intB)
    nB_cols = len(intB[0]) if nB_rows > 0 else 0

    if nA_cols != nB_rows:
        raise ValueError("Inner dimensions do not match for multiplication")

    # decide cutoff
    cut = cutoff if (cutoff is not None) else 1

    # pad to square power of two
    n = max(nA_rows, nA_cols, nB_rows, nB_cols)
    p = _next_power_of_two(n)
    A_pad = _pad_matrix(intA, p)
    B_pad = _pad_matrix(intB, p)

    # run recursive Strassen
    C_pad, muls, adds = _strassen_recursive(A_pad, B_pad, cut)

    # unpad to original output size (nA_rows x nB_cols)
    C_int = _unpad_matrix(C_pad, nA_rows, nB_cols)

    # convert back to digit-lists if inputs were digit matrices
    was_digit_input = _is_digit_matrix(A_in) or _is_digit_matrix(B_in)
    C_out = _to_digit_matrix(C_int) if was_digit_input else C_int

    elapsed = perf_counter() - start

    counters = {"muls": int(muls), "adds": int(adds), "time_s": float(elapsed)}
    if return_counters:
        return C_out, counters
    return C_out

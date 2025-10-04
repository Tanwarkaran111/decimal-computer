# decimal_computer/strassen.py
"""
Pure-Python Strassen backend (no NumPy / no Numba).

Exports:
    strassen_gemm(A, B, cutoff=None) -> C

This implementation:
 - converts entries to ints (accepts digit-lists / numeric strings / ints)
 - pads to power-of-two square matrices, applies Strassen recursively
 - falls back to schoolbook when below cutoff
 - returns nested lists of ints
"""
from __future__ import annotations
from typing import Any, List, Optional


def _entry_to_int(x: Any) -> int:
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (list, tuple)):
        try:
            return int("".join(str(int(d)) for d in x))
        except Exception:
            return int(x)
    return int(x)


def _ensure_int_matrix(M: List[List[Any]]) -> List[List[int]]:
    if not isinstance(M, list):
        raise ValueError("matrix must be a nested list")
    return [[_entry_to_int(cell) for cell in row] for row in M]


def _zero_matrix(n: int, m: Optional[int] = None) -> List[List[int]]:
    if m is None:
        m = n
    return [[0 for _ in range(m)] for _ in range(n)]


def _pad_to(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pad_matrix(A: List[List[int]], size: int) -> List[List[int]]:
    n = len(A)
    m = len(A[0]) if n else 0
    padded = _zero_matrix(size, size)
    for i in range(n):
        for j in range(m):
            padded[i][j] = A[i][j]
    return padded


def _unpad_matrix(A: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    return [row[:cols] for row in A[:rows]]


def _add(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(n)]


def _sub(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(n)]


def _split_matrix(A: List[List[int]]):
    n = len(A)
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    return A11, A12, A21, A22


def _join_quads(C11, C12, C21, C22):
    top = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bottom = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bottom


def _schoolbook_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    if n == 0:
        return []
    k = len(A[0])
    m = len(B[0])
    C = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for p in range(k):
            ap = A[i][p]
            if ap == 0:
                continue
            for j in range(m):
                C[i][j] += ap * B[p][j]
    return C


def _strassen_square(A: List[List[int]], B: List[List[int]], cutoff: int) -> List[List[int]]:
    n = len(A)
    if n <= cutoff:
        return _schoolbook_multiply(A, B)

    A11, A12, A21, A22 = _split_matrix(A)
    B11, B12, B21, B22 = _split_matrix(B)

    M1 = _strassen_square(_add(A11, A22), _add(B11, B22), cutoff)
    M2 = _strassen_square(_add(A21, A22), B11, cutoff)
    M3 = _strassen_square(A11, _sub(B12, B22), cutoff)
    M4 = _strassen_square(A22, _sub(B21, B11), cutoff)
    M5 = _strassen_square(_add(A11, A12), B22, cutoff)
    M6 = _strassen_square(_sub(A21, A11), _add(B11, B12), cutoff)
    M7 = _strassen_square(_sub(A12, A22), _add(B21, B22), cutoff)

    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    return _join_quads(C11, C12, C21, C22)


def strassen_gemm(A: List[List[Any]], B: List[List[Any]], cutoff: Optional[int] = None) -> List[List[int]]:
    a = _ensure_int_matrix(A)
    b = _ensure_int_matrix(B)

    n = len(a)
    if n == 0:
        return []
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("A must be rectangular")
    if k != len(b):
        raise ValueError("Inner dimension mismatch")
    m = len(b[0])
    if any(len(row) != m for row in b):
        raise ValueError("B must be rectangular")

    cutoff = int(cutoff) if cutoff is not None else 64
    S = max(n, k, m)
    P = _pad_to(S)

    A_pad = _pad_matrix(a, P)
    B_pad = _pad_matrix(b, P)

    C_pad = _strassen_square(A_pad, B_pad, cutoff=cutoff)
    C = _unpad_matrix(C_pad, n, m)
    return C

# decimal_computer/karatsuba.py
"""
Pure-Python 'karatsuba' backend (no NumPy / no Numba).

This module intentionally implements a correct, simple triple-loop
matrix multiply while keeping the name `multiply_matrices` for API compatibility.
It accepts entries as ints, digit-lists (MSB-first), or numeric strings.
"""
from __future__ import annotations
from typing import Any, List


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


def multiply_matrices(A: List[List[Any]], B: List[List[Any]]) -> List[List[int]]:
    """
    Multiply two matrices A (n x k) and B (k x m) and return a nested list of ints.
    This is a straightforward O(n*k*m) implementation (schoolbook).
    """
    a = _ensure_int_matrix(A)
    b = _ensure_int_matrix(B)

    n = len(a)
    if n == 0:
        return []
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("A must be rectangular")
    if len(b) != k:
        raise ValueError(f"Inner dimension mismatch: A is {n}x{k}, B is {len(b)}x{(len(b[0]) if b else 0)}")
    m = len(b[0])
    if any(len(row) != m for row in b):
        raise ValueError("B must be rectangular")

    C = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        ai = a[i]
        ci = C[i]
        for p in range(k):
            ap = ai[p]
            if ap == 0:
                continue
            bp = b[p]
            for j in range(m):
                ci[j] += ap * bp[j]
    return C

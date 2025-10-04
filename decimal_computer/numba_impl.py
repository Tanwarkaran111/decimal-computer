# decimal_computer/numba_impl.py
"""
Numba-backed matrix multiply wrapper.

Provides:
 - _has_numba: bool
 - multiply(A, B) -> either C_nested_list or (C_nested_list, counters_dict)

Counters dict has keys: 'muls', 'adds', 'time_s'
"""
from __future__ import annotations
from typing import Any, List, Tuple, Optional, Dict
import time

_has_numba = False
_numba_parallel = False

try:
    import numpy as _np
    import numba as _numba
    _has_numba = True
    try:
        # prefer njit with parallel if available
        @_numba.njit('int64[:, :](int64[:, :], int64[:, :])', parallel=False)
        def _numba_matmul(a, b):
            n, k = a.shape
            k2, m = b.shape
            C = _np.zeros((n, m), dtype=_np.int64)
            for i in range(n):
                for j in range(m):
                    s = 0
                    for p in range(k):
                        s += a[i, p] * b[p, j]
                    C[i, j] = s
            return C
    except Exception:
        # Fallback if specifying signature fails on some numba versions
        @_numba.njit()  # generic jit
        def _numba_matmul(a, b):
            n, k = a.shape
            k2, m = b.shape
            C = _np.zeros((n, m), dtype=_np.int64)
            for i in range(n):
                for j in range(m):
                    s = 0
                    for p in range(k):
                        s += a[i, p] * b[p, j]
                    C[i, j] = s
            return C

except Exception:
    _has_numba = False
    # we still import numpy for fallback
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # will error later if needed


def _to_numpy_int_matrix(M: List[List[Any]]):
    if _np is None:
        raise ImportError("numpy required for numba_impl")
    return _np.array([[int(x) for x in row] for row in M], dtype=_np.int64)


def _to_list(C_np) -> List[List[int]]:
    return [[int(x) for x in row] for row in C_np.tolist()]


def multiply(A: List[List[Any]], B: List[List[Any]]) -> Tuple[List[List[int]], Dict[str, Optional[float]]]:
    """
    Multiply two nested-list integer matrices.

    Returns:
      (C_nested_list, counters_dict)

    counters_dict contains:
      - 'muls' : estimated number of digit multiplications (n*n*n)
      - 'adds' : estimated number of adds (n*n*n)
      - 'time_s': elapsed seconds (float)
    """
    if not isinstance(A, list) or not isinstance(B, list):
        raise TypeError("A and B must be nested lists")

    # basic shape checks
    n = len(A)
    if n == 0:
        return [], {"muls": 0, "adds": 0, "time_s": 0.0}
    k = len(A[0])
    if k != len(B):
        raise ValueError("Inner dimension mismatch")

    # convert to numpy int64
    a_np = _to_numpy_int_matrix(A)
    b_np = _to_numpy_int_matrix(B)

    start = time.perf_counter()
    if _has_numba:
        # call compiled function
        C_np = _numba_matmul(a_np, b_np)
    else:
        # fallback pure-Python/NumPy dot (still returns correct result)
        # use numpy dot as fallback (fast in CPython)
        if _np is None:
            raise ImportError("numpy is required by numba_impl fallback")
        C_np = _np.dot(a_np, b_np).astype(_np.int64)
    elapsed = time.perf_counter() - start

    # counters: basic estimate (elementwise multiply-adds)
    n_rows, n_cols = C_np.shape
    muls = int(n_rows) * int(n_cols) * int(k)
    adds = muls  # rough estimate; could be muls - n_rows*n_cols but keep simple

    return _to_list(C_np), {"muls": muls, "adds": adds, "time_s": elapsed}


# Export symbol used by decimal_gemm
__all__ = ["multiply", "_has_numba"]

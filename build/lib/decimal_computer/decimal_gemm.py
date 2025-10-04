# decimal_computer/decimal_gemm.py
"""
Central GEMM entrypoint for the project.

Provides:
  - decimal_gemm_naive(A, B, ...) : main entry used by demos/tests
  - simple implementations / wrappers for schoolbook, karatsuba and strassen
Notes:
  - This file is defensive: it will try to call optimized helpers in other
    modules (if present) and otherwise fall back to safe Python/NumPy methods.
  - Keeps API used throughout the repo (reset_counters_before, return_counters, mul_algo, cutoff).
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

# try to import existing helpers if they exist in the package
try:
    # helpers that may be in decimal_digit_starter
    from .decimal_digit_starter import int_to_digits, digits_to_int  # type: ignore
except Exception:
    # provide fallbacks if digit helpers are not present (simple wrappers)
    def int_to_digits(x: int) -> List[int]:
        return [int(d) for d in str(abs(int(x)))]

    def digits_to_int(digs: List[int]) -> int:
        if not digs:
            return 0
        s = "".join(str(int(d)) for d in digs)
        return int(s)

# try to import strassen implementation (user provided in repo)
try:
    from .strassen import strassen_gemm  # type: ignore
except Exception:
    strassen_gemm = None  # will fallback to other options


logger = logging.getLogger("decimal_gemm")
logging.basicConfig(level=logging.INFO, format="[decimal_gemm] %(message)s")


# Simple counters container
class Counters:
    def __init__(self):
        self.muls = 0
        self.adds = 0

    def to_dict(self):
        return {"muls": int(self.muls), "adds": int(self.adds)}


def _to_numpy_int_matrix(M: List[List[Any]]) -> np.ndarray:
    """Convert a nested list matrix of ints/digits into a numpy int matrix."""
    return np.array([[int(x) for x in row] for row in M], dtype=np.int64)


def _from_numpy_int_matrix(A: np.ndarray) -> List[List[int]]:
    """Convert numpy int matrix -> nested list (python ints)"""
    return [[int(x) for x in row] for row in A.tolist()]


# -------------------------
# SCHOOLBOOK (fallback)
# -------------------------
def schoolbook_gemm(A: List[List[int]], B: List[List[int]], counters: Optional[Counters] = None) -> List[List[int]]:
    """Plain schoolbook multiply for small integer matrices (safe fallback)."""
    a = _to_numpy_int_matrix(A)
    b = _to_numpy_int_matrix(B)
    n, k = a.shape
    k2, m = b.shape
    assert k == k2, "inner dim mismatch"
    C = np.zeros((n, m), dtype=np.int64)
    # count ops exactly in elementwise multiply-add:
    for i in range(n):
        for j in range(m):
            s = 0
            for p in range(k):
                s += int(a[i, p]) * int(b[p, j])
                if counters:
                    counters.muls += 1
                    counters.adds += 1
            C[i, j] = s
    return _from_numpy_int_matrix(C)


# -------------------------
# KARATSUBA (very small fallback wrapper)
# -------------------------
def karatsuba_gemm(A: List[List[int]], B: List[List[int]], cutoff: Optional[int], counters: Optional[Counters] = None) -> List[List[int]]:
    """
    If you have a karatsuba implementation elsewhere, replace this wrapper.
    For now we use schoolbook but advertise karatsuba via the cutoff parameter.
    """
    # NOTE: to implement a real karatsuba on decimal-digit representations
    # you'd need digit-level functions. For now fallback to schoolbook.
    logger.debug("karatsuba_gemm called (cutoff=%s) - falling back to schoolbook", cutoff)
    return schoolbook_gemm(A, B, counters=counters)


# -------------------------
# STRASSEN wrapper
# -------------------------
def strassen_wrapper(A: List[List[int]], B: List[List[int]], cutoff: Optional[int], counters: Optional[Counters] = None) -> List[List[int]]:
    """
    Call into repo's strassen implementation if present, else fallback to schoolbook.
    Expected signature of strassen_gemm (if provided) is:
       strassen_gemm(A, B, cutoff=None) -> nested-list matrix
    """
    if strassen_gemm is None:
        logger.debug("strassen_gemm not available; falling back to schoolbook")
        return schoolbook_gemm(A, B, counters=counters)
    try:
        # assume strassen_gemm returns nested lists
        return strassen_gemm(A, B, cutoff=cutoff)
    except TypeError:
        # older signature maybe strassen_gemm(A, B)
        return strassen_gemm(A, B)


# -------------------------
# MAIN API
# -------------------------
def decimal_gemm_naive(
    A: List[List[int]],
    B: List[List[int]],
    reset_counters_before: bool = False,
    return_counters: bool = False,
    mul_algo: str = "auto",
    cutoff: Optional[int] = None,
) -> Any:
    """
    Main GEMM API used by examples/tests.

    Args:
      A, B: nested-list matrices of integers (or digit representations convertible to ints).
      reset_counters_before: ignored here (kept for API compatibility).
      return_counters: if True, returns (C, counters)
      mul_algo: "auto", "schoolbook", "karatsuba", "strassen"
      cutoff: optional cutoff (for karatsuba/strassen)
    Returns:
      If return_counters True: (C, counters_dict)
      else: C (nested list)
    """
    counters = Counters()
    start = time.perf_counter()

    # If requested 'auto', choose an algorithm heuristically based on matrix size
    if mul_algo == "auto":
        # tiny matrices: schoolbook; medium: karatsuba; larger: strassen if available
        try:
            n = len(A)
            if n <= 8:
                chosen = "schoolbook"
            elif n <= 64:
                chosen = "karatsuba"
            else:
                chosen = "strassen" if strassen_gemm is not None else "karatsuba"
        except Exception:
            chosen = "schoolbook"
        mul_algo = chosen
        logger.info("auto-selected algo=%r max_digit_len=%s cutoff=%s", mul_algo, None, cutoff)

    # Dispatch
    if mul_algo == "schoolbook":
        C = schoolbook_gemm(A, B, counters=counters)
    elif mul_algo == "karatsuba":
        C = karatsuba_gemm(A, B, cutoff=cutoff, counters=counters)
    elif mul_algo == "strassen":
        C = strassen_wrapper(A, B, cutoff=cutoff, counters=counters)
    else:
        raise ValueError(f"Unknown mul_algo: {mul_algo!r}")

    elapsed = time.perf_counter() - start

    # Logging: match the style used in your examples
    logger.info("%s result C = %s", mul_algo.upper(), C if (len(str(C)) < 200) else "[matrix]")
    logger.info("  digit muls: %d digit adds: %d time (s): %f", int(counters.muls), int(counters.adds), elapsed)

    if return_counters:
        return C, {"muls": int(counters.muls), "adds": int(counters.adds), "time_s": elapsed}
    else:
        return C


# Expose public symbol expected by __init__.py / tests
__all__ = ["decimal_gemm_naive", "schoolbook_gemm", "karatsuba_gemm", "strassen_wrapper"]

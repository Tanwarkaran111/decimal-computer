# phase2/gemm.py

import time

from phase2.parallel_karatsuba import karatsuba_gemm, _multiply_block_rows
from phase2.decimal_gemm import decimal_gemm_naive
from phase2.block_strassen import block_strassen

# Try importing Phase 3 auto selector
try:
    from phase3.auto_runtime import auto_gemm
    HAVE_AUTO = True
except ImportError:
    HAVE_AUTO = False

__all__ = ["gemm"]

# Known algorithm registry
_ALGOS = {
    "schoolbook": _multiply_block_rows,
    "karatsuba": karatsuba_gemm,
    "strassen": block_strassen,
    "decimal_naive": decimal_gemm_naive,
    # Optional: you can add "schoolbook_blocked" when you wire it in
}

def gemm(A, B, algo: str = None, digits: int = 1, debug: bool = False, time_it: bool = False):
    """
    Main GEMM entry point.

    Parameters
    ----------
    A, B : list[list[int]]
        Matrices to multiply
    algo : str or None
        If None → auto_gemm selector (Phase 3) if available, fallback otherwise
        If str → use that algorithm explicitly
    digits : int
        Number of digits (for decimal backends)
    debug : bool
        Print debug traces
    time_it : bool
        Print runtime in seconds

    Returns
    -------
    C : list[list[int]]
        Resulting matrix
    """

    t0 = time.perf_counter()

    if algo is None:
        if HAVE_AUTO:
            if debug:
                print(f"[phase2.gemm] Using auto_gemm selector for n={len(A)} digits={digits}")
            res = auto_gemm(A, B, digits=digits, debug=debug)
        else:
            if debug:
                print(f"[phase2.gemm] Using fallback: schoolbook _multiply_block_rows (n={len(A)} digits={digits})")
            res = _multiply_block_rows(A, B)
    else:
        func = _ALGOS.get(algo)
        if func is None:
            raise ValueError(f"Unknown algo: {algo} (known: {list(_ALGOS.keys())})")
        if debug:
            print(f"[phase2.gemm] Using explicit algo={algo} n={len(A)} digits={digits}")
        res = func(A, B)

    t1 = time.perf_counter()

    if time_it:
        print(f"[phase2.gemm] runtime: {t1 - t0:.6f} s")

    return res

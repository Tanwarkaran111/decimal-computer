#!/usr/bin/env python3
"""
hybrid_phase2.py
Adaptive hybrid selector (Phase2). Reads recommended_cutoffs.json if present.
Uses parallel_karatsuba.parallel_multiply and block_strassen.strassen_gemm.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

# Phase1 functions (fallbacks)
from decimal_computer.decimal_gemm import schoolbook_gemm, karatsuba_gemm, strassen_wrapper, decimal_gemm_naive

# Phase2 helpers
from phase2.parallel_karatsuba import parallel_multiply
from phase2.block_strassen import run_strassen

CUT_JSON = Path("D:/Invented_Library/recommended_cutoffs.json")
if CUT_JSON.exists():
    try:
        RECOMMENDED = json.loads(CUT_JSON.read_text())
    except Exception:
        RECOMMENDED = {}
else:
    RECOMMENDED = {}


def hybrid_gemm(A: List[List[Any]], B: List[List[Any]], digits: Optional[int] = None, size: Optional[int] = None,
                parallel_workers: Optional[int] = None, block_rows: int = 8, block_cutoff: int = 32, verbose: bool = False):
    """
    Hybrid:
      - Use schoolbook for tiny sizes
      - Use parallel_karatsuba.parallel_multiply for mid sizes
      - Use block-strassen (strassen_gemm with cutoff) for large sizes
    """
    n = size if size is not None else len(A)
    d = digits if digits is not None else (None)

    # If cutoffs known for this digit count, respect them
    if d is not None and str(d) in RECOMMENDED:
        rules = RECOMMENDED[str(d)]
        k_cut = rules.get("karatsuba")
        s_cut = rules.get("strassen")

        if k_cut is not None and n < int(k_cut):
            if verbose: print(f"[hybrid_phase2] Choosing schoolbook (n={n} < karatsuba_cut={k_cut})")
            return schoolbook_gemm(A, B)
        if s_cut is not None and n >= int(s_cut):
            if verbose: print(f"[hybrid_phase2] Choosing strassen (n={n} >= strassen_cut={s_cut})")
            return run_strassen(A, B, cutoff=block_cutoff, verbose=verbose)
        # otherwise choose karatsuba region
        if verbose: print(f"[hybrid_phase2] Choosing parallel karatsuba (n={n} >= karatsuba_cut)")
        return parallel_multiply(A, B, max_workers=parallel_workers, block_rows=block_rows, verbose=verbose)

    # Fallback heuristics
    if n < 8:
        if verbose: print(f"[hybrid_phase2] Fallback: schoolbook (n={n} < 8)")
        return schoolbook_gemm(A, B)
    if n < 64:
        if verbose: print(f"[hybrid_phase2] Fallback: parallel karatsuba (8<=n<{64})")
        return parallel_multiply(A, B, max_workers=parallel_workers, block_rows=block_rows, verbose=verbose)
    if verbose: print(f"[hybrid_phase2] Fallback: block-strassen (n>={64})")
    return run_strassen(A, B, cutoff=block_cutoff, verbose=verbose)


# simple convenience alias for tests
if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--digits", type=int, default=16)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--block-rows", type=int, default=8)
    p.add_argument("--block-cutoff", type=int, default=32)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # build matrices
    A = [[(i + j + 1) % 100 for j in range(args.n)] for i in range(args.n)]
    B = [[(i * j + 2) % 100 for j in range(args.n)] for i in range(args.n)]

    t0 = time.perf_counter()
    C = hybrid_gemm(A, B, digits=args.digits, size=args.n,
                    parallel_workers=args.workers,
                    block_rows=args.block_rows, block_cutoff=args.block_cutoff, verbose=args.verbose)
    t = time.perf_counter() - t0
    print(f"[hybrid_phase2] done elapsed={t:.6f}s, top-left:", [r[:2] for r in C[:2]])

#!/usr/bin/env python3
"""
parallel_karatsuba.py
Compute A x B by slicing A into row-blocks and multiplying each block in parallel
using the karatsuba_gemm wrapper (falls back to schoolbook if karatsuba backend missing).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import argparse
import os
import time

# Import karatsuba wrapper from your package
try:
    from decimal_computer.decimal_gemm import karatsuba_gemm, schoolbook_gemm, _matrix_to_ints
except Exception:
    # Best-effort fallback if package import style differs
    from decimal_computer import decimal_gemm as dg
    karatsuba_gemm = getattr(dg, "karatsuba_gemm", None)
    schoolbook_gemm = getattr(dg, "schoolbook_gemm")
    _matrix_to_ints = getattr(dg, "_matrix_to_ints", None)


def _multiply_block_rows(A, B, block_size: int = 16):
    """
    Blocked/tiled schoolbook multiply of A (n x p) and B (p x m).
    Pure-Python, tunable block_size (default 16).
    Returns C (n x m).
    """
    n = len(A)
    if n == 0:
        return []
    m = len(B[0])
    p = len(B)

    # allocate result
    C = [[0] * m for _ in range(n)]

    # block loops
    bs = int(block_size) if block_size > 0 else 16
    for ii in range(0, n, bs):
        iimax = min(ii + bs, n)
        for kk in range(0, p, bs):
            kkmax = min(kk + bs, p)
            for jj in range(0, m, bs):
                jjmax = min(jj + bs, m)
                for i in range(ii, iimax):
                    Ai = A[i]
                    Ci = C[i]
                    for k in range(kk, kkmax):
                        aik = Ai[k]
                        Bk = B[k]
                        # inner j loop
                        for j in range(jj, jjmax):
                            Ci[j] += aik * Bk[j]
    return C



def parallel_multiply(A: List[List], B: List[List], max_workers: int = None, block_rows: int = 8, verbose: bool = False) -> List[List]:
    """Parallel multiply by splitting A into row-blocks of block_rows each."""
    n = len(A)
    if n == 0:
        return []
    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 1)

    # Build list of row-slices
    slices = []
    for i in range(0, n, block_rows):
        slices.append((i, min(n, i + block_rows)))

    # Submit tasks
    results = [None] * len(slices)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for idx, (r0, r1) in enumerate(slices):
            A_block = A[r0:r1]
            fut = ex.submit(_multiply_block_rows, A_block, B)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
                if verbose:
                    print(f"[parallel] block {idx} done (rows {slices[idx][0]}..{slices[idx][1]-1})")
            except Exception as e:
                raise

    # Merge blocks into full C
    # determine m columns from result blocks
    C = []
    for block in results:
        C.extend(block)
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"[parallel] finished in {elapsed:.6f}s using {max_workers} workers (block_rows={block_rows})")
    return C


# CLI / quick test
def _make_test_matrix(n: int, val: int = 1):
    return [[(i + j + val) % 100 for j in range(n)] for i in range(n)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64, help="matrix size")
    p.add_argument("--block-rows", type=int, default=8, help="rows per task")
    p.add_argument("--workers", type=int, default=None, help="max workers")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    A = _make_test_matrix(args.n, 1)
    B = _make_test_matrix(args.n, 2)

    C = parallel_multiply(A, B, max_workers=args.workers, block_rows=args.block_rows, verbose=args.verbose)
    if args.verbose:
        print("Top-left 2x2 block:", [row[:2] for row in C[:2]])
    else:
        print("Done. Shape:", len(C), "x", (len(C[0]) if C else 0))

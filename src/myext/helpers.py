# src/myext/helpers.py  (REPLACE your existing file with this)
import os
import json
import time
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
# detect environment override once at import time
_force_threadpool = bool(int(os.environ.get("MYEXT_FORCE_THREADPOOL", "0")))

# compiled extension: try import, fall back to None
try:
    from . import gemm as _g  # compiled extension expected (gemm_openmp, gemm_block)
except Exception:
    _g = None

# env override to force threadpool (honored at call time)
_DEFAULT_FORCE_THREADPOOL = os.environ.get("MYEXT_FORCE_THREADPOOL", "0") == "1"

# default blocks (use tuned values after running autotune)
DEFAULT_BLOCKS = (64, 64, 32)

# cache path for autotune results (user home)
TUNE_CACHE = Path(os.environ.get("MYEXT_TUNE_CACHE", Path.home() / ".myext_tune.json"))


# paste near the top of src/myext/helpers.py (after imports)
# 1) Add this helper near the top of src/myext/helpers.py (after imports)

def _ensure_float64_c_contig(x, copy_if_needed=True):
    """
    Ensure x is a numpy.ndarray with dtype float64 and C-contiguous.
    Returns (arr, copied: bool).
    If copy_if_needed is False, raises ValueError when a conversion/copy would be required.
    """
    if not isinstance(x, np.ndarray):
        if not copy_if_needed:
            raise ValueError("requires ndarray float64 C-contiguous")
        return np.asarray(x, dtype=np.float64), True

    copied = False
    if x.dtype != np.float64:
        if not copy_if_needed:
            raise ValueError("requires dtype float64")
        x = x.astype(np.float64, copy=False)
        if x.dtype != np.float64:
            # fallback ensure
            x = x.astype(np.float64)
        copied = True

    if not x.flags.c_contiguous:
        if not copy_if_needed:
            raise ValueError("requires C-contiguous array")
        x = np.ascontiguousarray(x)
        copied = True

    return x, copied



def _ensure_dtype_and_c_contig(arr):
    """Return array as float64 and C-contiguous. Avoid copying if already OK."""
    if arr.dtype != np.float64 or not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr, dtype=np.float64)
    return arr

def gemm_threadpool(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                    blockM=64, blockN=64, blockK=64, workers=None):
    """Portable fallback using ThreadPoolExecutor and compiled gemm_block (releases GIL)."""
    # enforce types/layout for native kernels
    A = _ensure_dtype_and_c_contig(A)
    B = _ensure_dtype_and_c_contig(B)
    C = _ensure_dtype_and_c_contig(C)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    assert C.shape == (M, N)
    workers = workers or multiprocessing.cpu_count()

    nblocks_i = (M + blockM - 1) // blockM
    nblocks_j = (N + blockN - 1) // blockN
    nblocks_k = (K + blockK - 1) // blockK

    tasks = []
    # The compiled gemm_block expects arrays (C-contiguous) and block offsets.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i_block in range(nblocks_i):
            rowA = i_block * blockM
            for j_block in range(nblocks_j):
                colB = j_block * blockN
                for k_block in range(nblocks_k):
                    startK = k_block * blockK
                    # Submit native gemm_block which writes into C
                    if _g is None:
                        # fallback to small python multiply-add for the block
                        def _py_block(A=A, B=B, C=C, r=rowA, c=colB, k0=startK,
                                      bm=blockM, bn=blockN, bk=blockK):
                            r_end = min(r+bm, A.shape[0])
                            c_end = min(c+bn, B.shape[1])
                            k_end = min(k0+bk, A.shape[1])
                            C[r:r_end, c:c_end] += A[r:r_end, k0:k_end].dot(B[k0:k_end, c:c_end])
                        tasks.append(ex.submit(_py_block))
                    else:
                        tasks.append(ex.submit(_g.gemm_block, A, B, C, rowA, colB, startK,
                                               blockM, blockN, blockK))
        # wait for completion and propagate exceptions
        for fut in tasks:
            fut.result()

# 2) Replace your gemm(...) function body with this (keeps existing behavior, adds no_copy flag and logging)

def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray = None,
         blockM=None, blockN=None, blockK=None, workers=None, no_copy=False, verbose=False):
    """
    Public wrapper. Parameters:
      - A, B: input matrices
      - C: optional output buffer (must be shape (A.shape[0], B.shape[1]))
      - blockM/blockN/blockK: optional block size overrides
      - workers: threadpool size if used
      - no_copy: if True, raise ValueError when a copy would be needed
      - verbose: if True, print copy/log info
    """
    global _g

    # determine targets for block sizes
    bm = blockM if blockM is not None else DEFAULT_BLOCKS[0]
    bn = blockN if blockN is not None else DEFAULT_BLOCKS[1]
    bk = blockK if blockK is not None else DEFAULT_BLOCKS[2]

    # Ensure dtype & contiguity for A,B,C (raise if no_copy==True and a conversion/copy needed)
    try:
        A2, copiedA = _ensure_float64_c_contig(A, copy_if_needed=not no_copy)
        B2, copiedB = _ensure_float64_c_contig(B, copy_if_needed=not no_copy)
        if C is None:
            C2 = np.zeros((A2.shape[0], B2.shape[1]), dtype=np.float64)
            copiedC = True
        else:
            C2, copiedC = _ensure_float64_c_contig(C, copy_if_needed=not no_copy)
    except ValueError as e:
        # If user asked no_copy, propagate
        raise

    if verbose and (copiedA or copiedB or copiedC):
        print(f"[myext] copies created: A:{copiedA} B:{copiedB} C:{copiedC}")

    # Respect environment override to force threadpool (keeps existing behavior)
    if _force_threadpool:
        w = workers or multiprocessing.cpu_count()
        gemm_threadpool(A2, B2, C2, bm, bn, bk, w)
        return C2

    # Try native OpenMP path, fallback to threadpool on exception
    try:
        if _g is None:
            # if compiled extension is missing, fallback to threadpool
            raise RuntimeError("compiled backend not available")
        _g.gemm_openmp(A2, B2, C2, bm, bn, bk)
        return C2
    except Exception as e:
        if verbose:
            print("[myext] native gemm_openmp failed or not available; falling back to threadpool:", repr(e))
        w = workers or multiprocessing.cpu_count()
        gemm_threadpool(A2, B2, C2, bm, bn, bk, w)
        return C2


# ------------------------
# Autotune helpers
# ------------------------
def _time_run(N, bm, bn, bk, mode="openmp", omp_threads=8):
    """Time one run with given params. Returns elapsed, max_err."""
    # set env appropriately for the current process
    if mode == "openmp":
        os.environ.pop("MYEXT_FORCE_THREADPOOL", None)
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    else:
        os.environ["MYEXT_FORCE_THREADPOOL"] = "1"

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    # Warmup small block
    gemm(A[:64,:64], B[:64,:64], C[:64,:64], blockM=bm, blockN=bn, blockK=bk)

    t0 = time.perf_counter()
    gemm(A, B, C, blockM=bm, blockN=bn, blockK=bk)
    elapsed = time.perf_counter() - t0
    max_err = float(np.max(np.abs(C - A.dot(B))))
    return elapsed, max_err

def autotune_and_save(N=1024, candidates=None, threads=8, write_cache=True):
    if candidates is None:
        candidates = [
            (32,32,32),(64,64,64),(64,64,32),(32,64,32),(64,32,32),
            (128,64,32),(128,128,64)
        ]
    results = []
    for bm,bn,bk in candidates:
        try:
            t_open, err_open = _time_run(N, bm, bn, bk, mode="openmp", omp_threads=threads)
        except Exception:
            t_open, err_open = float("inf"), float("inf")
        try:
            t_thr, err_thr = _time_run(N, bm, bn, bk, mode="threadpool", omp_threads=threads)
        except Exception:
            t_thr, err_thr = float("inf"), float("inf")
        results.append({
            "blocks": (bm,bn,bk),
            "openmp_elapsed": t_open,
            "openmp_err": err_open,
            "threadpool_elapsed": t_thr,
            "threadpool_err": t_thr
        })
        print(f"tested {bm}x{bn}x{bk}: openmp={t_open:.4f}s thr={t_thr:.4f}s err_open={err_open:.4g}")

    valid = [r for r in results if r["openmp_err"] < 1e-6]
    if not valid:
        best = min(results, key=lambda r: r["openmp_elapsed"])
    else:
        best = min(valid, key=lambda r: r["openmp_elapsed"])

    best_blocks = tuple(best["blocks"])
    print("BEST (OpenMP) blocks:", best_blocks, "elapsed:", best["openmp_elapsed"])

    if write_cache:
        try:
            TUNE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(TUNE_CACHE, "w") as f:
                json.dump({"best_blocks": best_blocks, "results": results}, f, indent=2)
            print("Wrote tuning cache to", str(TUNE_CACHE))
        except Exception as e:
            print("Failed to write cache:", e)

    try:
        globals()["DEFAULT_BLOCKS"] = best_blocks
    except Exception:
        pass

    return best_blocks, results

def load_tuning_cache():
    try:
        if TUNE_CACHE.exists():
            with open(TUNE_CACHE, "r") as f:
                data = json.load(f)
            b = tuple(data.get("best_blocks", DEFAULT_BLOCKS))
            try:
                globals()["DEFAULT_BLOCKS"] = b
            except Exception:
                pass
            return b
    except Exception:
        pass
    return DEFAULT_BLOCKS

# try to load cached tuning on import (non-blocking; cheap)
load_tuning_cache()

# src/myext/helpers.py
import os
import json
import time
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# compiled extension
from . import gemm as _g  # myext.gemm (contains gemm_openmp and gemm_block)

# env override to force threadpool
_force_threadpool = os.environ.get("MYEXT_FORCE_THREADPOOL", "0") == "1"

# default blocks (use tuned values after running autotune)
DEFAULT_BLOCKS = (64, 64, 32)

# cache path for autotune results (user home)
TUNE_CACHE = Path(os.environ.get("MYEXT_TUNE_CACHE", Path.home() / ".myext_tune.json"))

def gemm_threadpool(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                    blockM=64, blockN=64, blockK=64, workers=None):
    """Portable fallback using ThreadPoolExecutor and compiled gemm_block (releases GIL)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    assert C.shape == (M, N)
    workers = workers or multiprocessing.cpu_count()

    nblocks_i = (M + blockM - 1) // blockM
    nblocks_j = (N + blockN - 1) // blockN
    nblocks_k = (K + blockK - 1) // blockK

    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i_block in range(nblocks_i):
            rowA = i_block * blockM
            for j_block in range(nblocks_j):
                colB = j_block * blockN
                for k_block in range(nblocks_k):
                    startK = k_block * blockK
                    tasks.append(ex.submit(_g.gemm_block, A, B, C, rowA, colB, startK,
                                           blockM, blockN, blockK))
        # wait for completion
        for fut in tasks:
            fut.result()

def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray = None,
         blockM=None, blockN=None, blockK=None, workers=None):
    """
    Public wrapper. Defaults come from DEFAULT_BLOCKS (which you can change or
    populate with autotune results). Force threadpool with MYEXT_FORCE_THREADPOOL=1.
    """
    if C is None:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    else:
        assert C.shape == (A.shape[0], B.shape[1])

    # choose block sizes (explicit args > cached/default)
    bm = blockM if blockM is not None else DEFAULT_BLOCKS[0]
    bn = blockN if blockN is not None else DEFAULT_BLOCKS[1]
    bk = blockK if blockK is not None else DEFAULT_BLOCKS[2]

    if _force_threadpool:
        w = workers or multiprocessing.cpu_count()
        gemm_threadpool(A, B, C, bm, bn, bk, w)
        return C

    # Try OpenMP path and fallback to threadpool on exception
    try:
        _g.gemm_openmp(A, B, C, bm, bn, bk)
        return C
    except Exception:
        w = workers or multiprocessing.cpu_count()
        gemm_threadpool(A, B, C, bm, bn, bk, w)
        return C

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
    """
    Run small autotune across candidate block tuples. Returns best_openmp tuple.
    - N: matrix size used for timing (default 1024)
    - candidates: list of (bm,bn,bk) tuples. If None, uses sane defaults.
    - threads: OMP thread count to test.
    - write_cache: whether to persist result to TUNE_CACHE.
    """
    if candidates is None:
        candidates = [
            (32,32,32),(64,64,64),(64,64,32),(32,64,32),(64,32,32),
            (128,64,32),(128,128,64)
        ]
    results = []
    for bm,bn,bk in candidates:
        try:
            t_open, err_open = _time_run(N, bm, bn, bk, mode="openmp", omp_threads=threads)
        except Exception as e:
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
            "threadpool_err": err_thr
        })
        # print progress
        print(f"tested {bm}x{bn}x{bk}: openmp={t_open:.4f}s thr={t_thr:.4f}s err_open={err_open:.4g}")

    # choose best openmp by elapsed (with acceptable error)
    valid = [r for r in results if r["openmp_err"] < 1e-6]
    if not valid:
        # fallback: choose by elapsed, ignoring error
        best = min(results, key=lambda r: r["openmp_elapsed"])
    else:
        best = min(valid, key=lambda r: r["openmp_elapsed"])

    best_blocks = tuple(best["blocks"])
    print("BEST (OpenMP) blocks:", best_blocks, "elapsed:", best["openmp_elapsed"])

    # persist
    if write_cache:
        try:
            TUNE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(TUNE_CACHE, "w") as f:
                json.dump({"best_blocks": best_blocks, "results": results}, f, indent=2)
            print("Wrote tuning cache to", str(TUNE_CACHE))
        except Exception as e:
            print("Failed to write cache:", e)

    # apply immediately for current process (set module-level DEFAULT_BLOCKS)
    try:
        globals()["DEFAULT_BLOCKS"] = best_blocks
    except Exception:
        pass

    return best_blocks, results


def load_tuning_cache():
    """Load tuning cache if present and set DEFAULT_BLOCKS."""
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

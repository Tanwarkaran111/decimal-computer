# bench/bench_gemm.py
"""
Benchmark our AVX2 micro-kernel vs NumPy BLAS.
This script:
  1. Loads our compiled native GEMM dispatch function (via ctypes)
  2. Runs timed multiplications for increasing matrix sizes
  3. Computes GFLOPS and compares with NumPy's dot()
"""

import os
import time
import ctypes
import numpy as np
from pathlib import Path

# --- Locate compiled shared library (.dll or .so)
build_dir = Path(__file__).resolve().parent.parent / "build"
libname = "_native.so" if os.name != "nt" else "test_micro_kernel.exe"  # we'll load the .exe for now
libpath = build_dir / libname

# We compiled test_micro_kernel.exe earlier; for benchmarking, recompile soon as a DLL.
# For now we reuse gemm_dispatch from kernel_wrapper.c (compiled into .exe)

# --- Define ctypes prototype for gemm_dispatch
# When we recompile as DLL, these signatures will still apply.
dllpath = build_dir / "gemm_native.dll"
_lib = ctypes.CDLL(str(dllpath))

_lib.gemm_dispatch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int
]
_lib.gemm_dispatch.restype = None


def gemm_native(A, B, use_avx2=True):
    """Call our native GEMM wrapper"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float64)
    _lib.gemm_dispatch(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        C.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_int(K), ctypes.c_int(N), ctypes.c_int(N),
        ctypes.c_int(1 if use_avx2 else 0),
    )
    return C


def benchmark_once(M, N, K, reps=3):
    """Run timing comparison for our native vs numpy"""
    A = np.random.randn(M, K)
    B = np.random.randn(K, N)

    # Warm-up
    gemm_native(A, B)
    np.dot(A, B)

    # Native timing
    t0 = time.perf_counter()
    for _ in range(reps):
        gemm_native(A, B)
    t_native = (time.perf_counter() - t0) / reps

    # NumPy timing
    t0 = time.perf_counter()
    for _ in range(reps):
        np.dot(A, B)
    t_numpy = (time.perf_counter() - t0) / reps

    gflops_native = (2 * M * N * K) / (t_native * 1e9)
    gflops_numpy = (2 * M * N * K) / (t_numpy * 1e9)

    print(f"{M:5d}x{K:<5d} * {K:5d}x{N:<5d} | "
          f"Our AVX2: {gflops_native:8.2f} GFLOPS | NumPy: {gflops_numpy:8.2f} GFLOPS")

    return gflops_native, gflops_numpy


def main():
    print("=== GEMM Micro-Kernel Benchmark (AVX2 vs NumPy) ===")
    sizes = [64, 128, 256, 512, 1024]
    results = []
    for sz in sizes:
        results.append(benchmark_once(sz, sz, sz))
    print("Done.")

if __name__ == "__main__":
    main()

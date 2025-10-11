# benchmark_gemm.py
import os
import time
import ctypes
import numpy as np
from statistics import median

# === CONFIG ===
DLL_PATH = r"D:\Invented_library\build\gemm_native_packed_final3.dll"
UCRT_BIN = r"D:\ucrt64\bin"     # ensure runtime dependencies are found
FUNC_NAMES = ["gemm_native_packed_final3",
              "gemm_native_public_simple",
              "gemm_native_public"]  # try several possible exported names

SIZES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]
# number of timed repeats (we take median)
REPEATS = 5
# number of warm-up calls before timing
WARMUPS = 2

# === helper functions ===
def load_gemm_func():
    # make sure runtime DLLs are found
    try:
        os.add_dll_directory(UCRT_BIN)
    except Exception:
        pass

    lib = ctypes.CDLL(DLL_PATH)
    for name in FUNC_NAMES:
        try:
            f = getattr(lib, name)
        except AttributeError:
            continue
        # signature: (void*,void*,void*, int,int,int, int,int,int) -> int
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int]
        f.restype = ctypes.c_int
        print(f"[INFO] Loaded function '{name}' from {DLL_PATH}")
        return f, name
    raise RuntimeError(f"No known entrypoint found in {DLL_PATH}. Tried: {FUNC_NAMES}")

def time_func(func, A, B, C, M, N, K, lda, ldb, ldc, repeats=5, warmups=2):
    # Warmups
    for _ in range(warmups):
        rc = func(A.ctypes.data_as(ctypes.c_void_p),
                  B.ctypes.data_as(ctypes.c_void_p),
                  C.ctypes.data_as(ctypes.c_void_p),
                  M, N, K, lda, ldb, ldc)
        if rc == 0:
            raise RuntimeError("Native GEMM returned 0 (error) during warmup")

    times = []
    for _ in range(repeats):
        C.fill(0.0)
        t0 = time.perf_counter()
        rc = func(A.ctypes.data_as(ctypes.c_void_p),
                  B.ctypes.data_as(ctypes.c_void_p),
                  C.ctypes.data_as(ctypes.c_void_p),
                  M, N, K, lda, ldb, ldc)
        t1 = time.perf_counter()
        if rc == 0:
            raise RuntimeError("Native GEMM returned 0 (error) during timing")
        times.append(t1 - t0)
    return median(times)

def time_numpy(A, B, repeats=5, warmups=2):
    # Warmups
    for _ in range(warmups):
        _ = A @ B
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return median(times)

def gflops_from_time(M, N, K, t):
    # GEMM flops ~ 2*M*N*K
    ops = 2.0 * M * N * K
    return ops / (t * 1e9)
  # try 4 or even 2

def run_bench():
    gemm_func, func_name = load_gemm_func()

    print("Size     |  native time (s)  GFLOPS  |  numpy time (s)  GFLOPS  |  correctness")
    print("-" * 96)
    for (M, N, K) in SIZES:
        # Prepare row-major (C-order) data (kernel expects row-major)
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float64))
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float64))
        C_native = np.ascontiguousarray(np.zeros((M, N), dtype=np.float64))

        # Leading dims: for row-major, we pass lda=K, ldb=N, ldc=N to match how you tested.
        lda = K
        ldb = N
        ldc = N

        # Time native
        t_native = time_func(gemm_func, A, B, C_native, M, N, K, lda, ldb, ldc,
                             repeats=REPEATS, warmups=WARMUPS)
        gfl_native = gflops_from_time(M, N, K, t_native)

        # Time NumPy
        t_numpy = time_numpy(A, B, repeats=REPEATS, warmups=WARMUPS)
        gfl_numpy = gflops_from_time(M, N, K, t_numpy)

        # Correctness: compare with numpy result
        C_ref = A @ B
        max_abs_diff = float(np.max(np.abs(C_ref - C_native)))

        print(f"{M:4d}x{N:4d}x{K:4d} | {t_native:12.6f}  {gfl_native:7.2f} | "
              f"{t_numpy:12.6f}  {gfl_numpy:7.2f} | {max_abs_diff:.3e}")

if __name__ == "__main__":
    run_bench()

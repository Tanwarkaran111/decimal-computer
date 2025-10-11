#!/usr/bin/env python3
import numpy as np, time, ctypes, os, argparse, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dll", type=str, default=r"D:\Invented_library\build\gemm_native_packed.dll")
    args = p.parse_args()

    os.add_dll_directory(r"D:\ucrt64\bin")
    try:
        lib = ctypes.CDLL(args.dll)
    except OSError as e:
        print(f"ERROR: Failed to load DLL '{args.dll}': {e}", file=sys.stderr)
        sys.exit(2)

    M,N,K = args.M, args.N, args.K
    A = np.random.randn(M, K).astype(np.float64)
    B = np.random.randn(K, N).astype(np.float64)
    C = np.zeros((M, N), dtype=np.float64)

    lib.gemm_dispatch_packed.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.gemm_dispatch_packed.restype = None

    # Warmup runs
    for i in range(args.warmup):
        lib.gemm_dispatch_packed(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            M, N, K,
            K, N, N
        )

    times = []
    for i in range(args.repeats):
        start = time.perf_counter()
        lib.gemm_dispatch_packed(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            M, N, K,
            K, N, N
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    best = min(times)
    gflops = (2.0 * M * N * K) / (best * 1e9)

    # CSV-friendly output
    print(f"Peak GFLOPS = {gflops:.2f}  (time={best:.3f}s)")
    # also emit a single-line CSV with extras
    print(f"{M},{N},{K},{args.repeats},{args.warmup},{best:.6f},{gflops:.6f}")

if __name__ == "__main__":
    main()

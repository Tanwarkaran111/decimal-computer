#!/usr/bin/env python3
"""
size_sweep.py

Run a sweep of GEMM sizes against your DLL and report timing + max absolute error.

Usage examples:
  python size_sweep.py --dll build/gemm_native_packed_final3.dll --sizes 64,128,256
  python size_sweep.py --dll build/gemm_native_packed_final3.dll --sizes 64-512:64 --fixedK 64
  python size_sweep.py --dll build/gemm_native_packed_final3.dll --triples 64,64,64;128,128,128 --outfile results.csv
"""

import os
import sys
import argparse
import ctypes
import numpy as np
import time
import csv
from pathlib import Path

# ---------- Helper: parse size specs ----------
def parse_sizes_arg(s: str):
    """Parse sizes argument.
    Supported forms:
      "64,128,256" -> [64,128,256]
      "64-512:64"  -> range(64,513,64)
    """
    s = s.strip()
    out = []
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            # range like 64-512:64 (optional step)
            if ':' in part:
                rng, step = part.split(':', 1)
                start, end = rng.split('-', 1)
                start, end, step = int(start), int(end), int(step)
            else:
                start, end = part.split('-', 1)
                start, end, step = int(start), int(end), 1
            out.extend(list(range(start, end+1, step)))
        else:
            out.append(int(part))
    return out

def parse_triples_arg(s: str):
    """Parse triples argument like '64,64,64;128,128,128' -> list of (M,N,K)."""
    if not s:
        return []
    triples = []
    for tok in s.split(';'):
        parts = [int(x) for x in tok.strip().split(',') if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid triple: {tok}")
        triples.append(tuple(parts))
    return triples

# ---------- DLL loader ----------
class GemmDll:
    def __init__(self, dll_path):
        self.dll_path = str(dll_path)
        if not Path(self.dll_path).exists():
            raise FileNotFoundError(f"DLL not found: {self.dll_path}")
        # On Windows ensure ucrt path available (best-effort)
        try:
            os.add_dll_directory(r"D:\ucrt64\bin")
        except Exception:
            pass
        self.lib = ctypes.CDLL(self.dll_path)
        # Try to find wrapper symbol first (int args)
        self.fn = None
        self.is_wrapper = False
        if hasattr(self.lib, "gemm_native_packed_final3"):
            self.fn = self.lib.gemm_native_packed_final3
            # signature: int gemm_native_packed_final3(double* A, double* B, double* C,
            #                                        int M,int N,int K, int lda,int ldb,int ldc)
            self.fn.restype = ctypes.c_int
            self.fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            self.is_wrapper = True
        elif hasattr(self.lib, "gemm_dispatch_packed"):
            # gemm_dispatch_packed(const void* A, const void* B, void* C,
            #                      size_t M,size_t N,size_t K,
            #                      size_t lda,size_t ldb,size_t ldc)
            self.fn = self.lib.gemm_dispatch_packed
            self.fn.restype = None
            # use c_size_t for sizes
            self.fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
            ]
            self.is_wrapper = False
        else:
            raise RuntimeError("DLL does not export gemm_native_packed_final3 or gemm_dispatch_packed")

    def call(self, A, B, C, M, N, K, lda=None, ldb=None, ldc=None):
        if lda is None: lda = K
        if ldb is None: ldb = N
        if ldc is None: ldc = N
        if self.is_wrapper:
            return self.fn(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(M), int(N), int(K),
                int(lda), int(ldb), int(ldc)
            )
        else:
            # gemm_dispatch_packed returns void
            return self.fn(
                A.ctypes.data_as(ctypes.c_void_p),
                B.ctypes.data_as(ctypes.c_void_p),
                C.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(M), ctypes.c_size_t(N), ctypes.c_size_t(K),
                ctypes.c_size_t(lda), ctypes.c_size_t(ldb), ctypes.c_size_t(ldc)
            )

# ---------- Single test run ----------
def run_once(gemm: GemmDll, M, N, K, lda=None, ldb=None, ldc=None, warmups=1):
    # create deterministic random matrices
    A = np.random.rand(M, K).astype(np.float64)
    B = np.random.rand(K, N).astype(np.float64)
    C = np.zeros((M, N), dtype=np.float64)
    # reference
    C_ref = A @ B

    # warmup
    for _ in range(warmups):
        gemm.call(A, B, C, M, N, K, lda, ldb, ldc)
        # zero C again to measure only one invocation
        C.fill(0.0)

    # timed call
    t0 = time.perf_counter()
    gemm.call(A, B, C, M, N, K, lda, ldb, ldc)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    max_err = float(np.abs(C - C_ref).max())
    # GFLOP/s: 2*M*N*K ops
    gflops = (2.0 * M * N * K) / (elapsed * 1e9) if elapsed > 0 else float('inf')
    return {
        "M": M, "N": N, "K": K,
        "time_s": elapsed, "gflops": gflops,
        "max_abs_error": max_err
    }

# ---------- Main CLI ----------
def main():
    p = argparse.ArgumentParser(description="Sweep GEMM sizes against DLL and report time + error")
    p.add_argument("--dll", "-d", default="build/gemm_native_packed_final3.dll", help="Path to DLL")
    p.add_argument("--sizes", help="Comma or range list of sizes for M,N,K (e.g. 64,128 or 64-256:64)")
    p.add_argument("--fixedK", type=int, default=None, help="Fix K to a value (otherwise K=M)")
    p.add_argument("--triples", help="Explicit triples as M,N,K;... e.g. 64,64,64;128,128,128")
    p.add_argument("--outfile", "-o", help="Write CSV to file (otherwise stdout)")
    p.add_argument("--warmups", type=int, default=1, help="Number of warmup calls before timing")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducible matrices")
    p.add_argument("--threads", type=int, default=1, help="Set OMP_NUM_THREADS for deterministic runs")
    args = p.parse_args()

    np.random.seed(args.seed)
    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    dll_path = Path(args.dll)
    if not dll_path.exists():
        print(f"ERROR: DLL not found: {dll_path}", file=sys.stderr)
        sys.exit(2)

    gemm = GemmDll(dll_path)

    triples = []
    if args.triples:
        triples = parse_triples_arg(args.triples)
    elif args.sizes:
        sizes = parse_sizes_arg(args.sizes)
        if not sizes:
            print("No sizes parsed, exiting", file=sys.stderr); sys.exit(1)
        for s in sizes:
            K = args.fixedK if args.fixedK is not None else s
            triples.append((s, s, K))
    else:
        # default sizes
        default_sizes = [32, 64, 128, 256]
        for s in default_sizes:
            K = args.fixedK if args.fixedK is not None else s
            triples.append((s, s, K))

    # Output CSV header
    out_rows = []
    header = ["M", "N", "K", "time_s", "gflops", "max_abs_error"]
    for (M, N, K) in triples:
        print(f"Running M={M}, N={N}, K={K} ...", flush=True)
        try:
            res = run_once(gemm, M, N, K, lda=None, ldb=None, ldc=None, warmups=args.warmups)
        except Exception as e:
            print(f"ERROR running {M},{N},{K}: {e}", file=sys.stderr)
            res = {"M": M, "N": N, "K": K, "time_s": None, "gflops": None, "max_abs_error": None}
        out_rows.append(res)
        # print immediate summary
        print(" -> time = {:.6f}s, gflops = {:.3f}, max_err = {:.6e}".format(
            res["time_s"] if res["time_s"] is not None else float('nan'),
            res["gflops"] if res["gflops"] is not None else float('nan'),
            res["max_abs_error"] if res["max_abs_error"] is not None else float('nan')
        ), flush=True)

    # Write CSV
    if args.outfile:
        with open(args.outfile, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in out_rows:
                writer.writerow({k: r.get(k) for k in header})
        print(f"Wrote results to {args.outfile}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=header)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k) for k in header})

if __name__ == "__main__":
    main()

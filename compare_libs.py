# compare_libs.py
"""
Compare myext.gemm against NumPy (and optional backends if installed).
Saves CSV results and prints a short table.

Usage:
    # set threads for OpenMP builds (optional)
    # PowerShell:
    $env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
    $env:OMP_NUM_THREADS="8"
    python compare_libs.py

The script will:
 - detect available backends
 - run warmups and timed repeats
 - report elapsed time (avg), GFLOPS, max absolute error and max relative error
 - save bench_results.csv
"""

import time
import csv
import importlib
from pathlib import Path
import numpy as np
import os
import math

# Try to import our compiled extension
try:
    # ensure local src is importable (user should set PYTHONPATH to src when running)
    from myext import gemm as myext_gemm_module
    from myext.helpers import gemm as myext_gemm
    HAVE_MYEXT = True
except Exception as e:
    myext_gemm = None
    myext_gemm_module = None
    HAVE_MYEXT = False

# Backends to test
backends = []

# NumPy baseline (required)
backends.append(("numpy", "dot"))

# Optional: SciPy BLAS (if installed)
try:
    import scipy
    from scipy.linalg import blas as scipy_blas
    backends.append(("scipy_blas", "dgemm"))
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Optional: PyTorch (if installed)
try:
    import torch
    backends.append(("torch", "matmul"))
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# Optional: myext
if HAVE_MYEXT:
    backends.append(("myext", "gemm"))

# Configs
N = 1024            # matrix size (change if memory/time is an issue)
REPEATS = 5         # how many timed runs to average
WARMUPS = 1
BLOCKS_DEFAULT = None  # leave None so myext uses its DEFAULT_BLOCKS

# Prepare random matrices (double precision)
rng = np.random.default_rng(12345)
A = rng.random((N, N), dtype=np.float64)
B = rng.random((N, N), dtype=np.float64)
# We'll use NumPy result as reference
C_ref = A.dot(B)

def run_numpy(A, B):
    return A.dot(B)

def run_scipy(A, B):
    # scipy.linalg.blas.dgemm: C = alpha*A*B + beta*C  (expects Fortran-ordered arrays often)
    # ensure shapes and ordering
    # dgemm(a, b, c=None, alpha=1.0, beta=0.0, trans_a=False, trans_b=False, overwrite_c=False)
    a = np.asfortranarray(A)
    b = np.asfortranarray(B)
    c = scipy_blas.dgemm(alpha=1.0, a=a, b=b, c=None, overwrite_c=0)
    return np.array(c)

def run_torch(A, B):
    # convert to torch double tensors on CPU
    tA = torch.from_numpy(A)
    tB = torch.from_numpy(B)
    tC = torch.matmul(tA, tB)
    return tC.numpy()

def run_myext(A, B, blocks=None):
    # myext.gemm modifies C in-place
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    if blocks is None:
        myext_gemm(A, B, C)     # uses defaults from helpers
    else:
        myext_gemm(A, B, C, blockM=blocks[0], blockN=blocks[1], blockK=blocks[2])
    return C

def timed_run(func, A, B, repeats=3, warmups=1):
    # warmups
    for _ in range(warmups):
        _ = func(A, B)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = func(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times

def max_abs_err(C, Creference):
    return float(np.max(np.abs(C - Creference)))

def max_rel_err(C, Creference):
    denom = np.max(np.abs(Creference))
    if denom == 0:
        return float(np.max(np.abs(C - Creference)))
    return float(np.max(np.abs(C - Creference)) / denom)

def flop_count(N):
    # GEMM count ~ 2*N^3 floating-point ops
    return 2.0 * (N ** 3)

def gflops_from_time(seconds, N):
    return flop_count(N) / (seconds * 1e9)

# Build runners
runners = []
for name, tag in backends:
    if name == "numpy":
        runners.append(("numpy", run_numpy))
    elif name == "scipy_blas" and HAVE_SCIPY:
        runners.append(("scipy_blas", run_scipy))
    elif name == "torch" and HAVE_TORCH:
        runners.append(("torch", run_torch))
    elif name == "myext" and HAVE_MYEXT:
        # prefer using tuned blocks if available
        def runner_myext(a, b):
            return run_myext(a, b, blocks=BLOCKS_DEFAULT)
        runners.append(("myext", runner_myext))

if not runners:
    print("No backends found (need at least numpy). Exiting.")
    raise SystemExit(1)

results = []
print("Backends to test:", [r[0] for r in runners])
print(f"Matrix size: {N}x{N}  repeats={REPEATS}  warmups={WARMUPS}")

for name, func in runners:
    print("\n== Testing", name)
    times = timed_run(func, A, B, repeats=REPEATS, warmups=WARMUPS)
    avg = sum(times) / len(times)
    best = min(times)
    worst = max(times)
    gfl = gflops_from_time(avg, N)
    # correctness: compute result once more to compare to reference
    try:
        C = func(A, B)
    except Exception as e:
        print(f"  ERROR running {name}: {e}")
        continue
    abs_err = max_abs_err(C, C_ref)
    rel_err = max_rel_err(C, C_ref)
    print(f"  times: {times}")
    print(f"  avg {avg:.4f}s  best {best:.4f}s  worst {worst:.4f}s  {gfl:.2f} GFLOPS")
    print(f"  max_abs_err={abs_err:.6g}  max_rel_err={rel_err:.6g}")
    results.append({
        "backend": name,
        "avg_s": avg,
        "best_s": best,
        "worst_s": worst,
        "gflops": gfl,
        "max_abs_err": abs_err,
        "max_rel_err": rel_err,
        "times": times
    })

# Save CSV
out_csv = Path("compare_results.csv")
with out_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["backend","avg_s","best_s","worst_s","gflops","max_abs_err","max_rel_err","times"])
    for r in results:
        writer.writerow([r["backend"], r["avg_s"], r["best_s"], r["worst_s"], r["gflops"], r["max_abs_err"], r["max_rel_err"], ";".join(map(str,r["times"]))])

print("\nWrote", out_csv)
print("Done.")

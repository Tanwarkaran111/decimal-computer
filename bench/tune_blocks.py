# bench/tune_blocks.py
"""
Auto-tune BLOCK_M, BLOCK_N, BLOCK_K by rebuilding gemm_native_packed.dll and running bench_gemm.py.

Run this from MSYS2 UCRT64 (so `gcc` is on PATH). It calls:
  gcc ... -DBLOCK_M=... -DBLOCK_N=... -DBLOCK_K=...
then runs bench_gemm.py and parses GFLOPS lines.

Adjust GRID lists to expand/contract search.
"""

import subprocess
import sys
import re
from pathlib import Path
import time
import csv

PYTHON_EXE = r"D:\ucrt64\bin\python.exe"
ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
SRC = ROOT / "src"
INCLUDE = ROOT / "include"
BENCH = ROOT / "bench"
GCC = "gcc"  # assume in PATH (MSYS2 UCRT64)

# Candidate grids (start conservative; you can expand later)
BLOCK_M_list = [96, 120, 144, 192]   # MR multiples (tuned to L2)
BLOCK_N_list = [128, 192, 256, 320]  # choose multiples of NR=8
BLOCK_K_list = [32, 48, 64, 96]      # small K blocks to fit cache

# Path to bench script
BENCH_SCRIPT = BENCH / "bench_gemm.py"

# Output CSV
OUT_CSV = BENCH / "tune_results.csv"

# Command template to build DLL
GCC_CMD_BASE = [
    GCC, "-O3", "-mavx2", "-mfma", "-std=c11", "-shared", "-fPIC", "-fopenmp",
    "-I", str(INCLUDE),
    str(SRC / "micro_kernel_scalar.c"),
    str(SRC / "micro_kernel_avx2.c"),
    str(SRC / "micro_kernel_avx2_opt.c"),
    str(SRC / "kernel_wrapper_packed.c"),
    "-o", str(BUILD / "gemm_native_packed.dll"),
    "-lm"
]


# Regex to capture GFLOPS from bench output lines (our bench prints lines like: " 256x256 ... | Our AVX2:   40.00 GFLOPS | NumPy:  xx")
GFLOPS_RE = re.compile(r"([0-9]+\.[0-9]+)\s*GFLOPS", re.IGNORECASE)


def build_with_flags(BM, BN, BK):
    compile_flags = ["-D", f"BLOCK_M={BM}", "-D", f"BLOCK_N={BN}", "-D", f"BLOCK_K={BK}"]
    cmd = GCC_CMD_BASE[:]
    # insert defines right after gcc base (before source files)
    # We'll put them at start (gcc accepts -D anywhere)
    cmd = [GCC] + compile_flags + GCC_CMD_BASE[1:]
    print("BUILD CMD:", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, cwd=str(BUILD))
    return p.returncode == 0

def run_bench():
    # run bench_gemm.py and capture output (single run)
    p = subprocess.run([PYTHON_EXE, str(BENCH_SCRIPT)], cwd=str(ROOT), capture_output=True, text=True)
    out = p.stdout + p.stderr
    return out

def parse_best_gflops(output):
    # Normalize line endings
    output = output.replace('\r', '\n')
    vals = GFLOPS_RE.findall(output)
    if not vals:
        print("⚠️  No GFLOPS found in output!")
        print(output[:500])  # show first 500 chars for debugging
        return 0.0
    nums = [float(v) for v in vals]
    # Take the largest value as representative performance
    best = max(nums)
    return best


def main():
    results = []
    total = len(BLOCK_M_list) * len(BLOCK_N_list) * len(BLOCK_K_list)
    i = 0
    for BM in BLOCK_M_list:
        for BN in BLOCK_N_list:
            for BK in BLOCK_K_list:
                i += 1
                print(f"\n[{i}/{total}] Testing BLOCK_M={BM} BLOCK_N={BN} BLOCK_K={BK}")
                t0 = time.time()
                ok = build_with_flags(BM, BN, BK)
                if not ok:
                    print("Build failed for", BM, BN, BK)
                    continue
                out = run_bench()
                gflops = parse_best_gflops(out)
                elapsed = time.time() - t0
                print(f"Result: {gflops:.2f} GFLOPS (elapsed {elapsed:.1f}s)")
                results.append((BM, BN, BK, gflops, elapsed))
                # Save incremental results to CSV
                with open(OUT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([BM, BN, BK, f"{gflops:.6f}", f"{elapsed:.2f}"])
    # Print best result
    if results:
        best = max(results, key=lambda r: r[3])
        print("\nBEST CONFIG:", best)
    else:
        print("No successful builds/benchmarks recorded.")

if __name__ == "__main__":
    # write header
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["BLOCK_M", "BLOCK_N", "BLOCK_K", "GFLOPS", "SECONDS"])
    main()

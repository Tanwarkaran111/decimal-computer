# bench_openmp_vs_threadpool.py
import os, time, csv, sys
import numpy as np
from pathlib import Path

# ensure local package import
sys.path.insert(0, str(Path.cwd() / "src"))
from myext import gemm

def run_once(N, blockM, blockN, blockK, mode, omp_threads=None):
    # mode: "openmp" or "threadpool"
    env = os.environ.copy()
    if mode == "openmp":
        if omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(omp_threads)
        env.pop("MYEXT_FORCE_THREADPOOL", None)
    else:
        env["MYEXT_FORCE_THREADPOOL"] = "1"
    # set env for current process (affects C runtime if set before import/call)
    os.environ.update({k: v for k, v in env.items() if k in ("OMP_NUM_THREADS","MYEXT_FORCE_THREADPOOL")})
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)
    # warmup
    gemm(A[:64,:64], B[:64,:64], C[:64,:64], blockM=blockM, blockN=blockN, blockK=blockK)
    t0 = time.time()
    gemm(A, B, C, blockM=blockM, blockN=blockN, blockK=blockK)
    elapsed = time.time() - t0
    # sanity check
    max_err = np.max(np.abs(C - A.dot(B)))
    return elapsed, max_err

def main():
    sizes = [512, 1024]      # matrix sizes to test
    blocks = [(64,64,64), (32,32,32), (128,128,64)]
    threads = [1, 2, 4, 8]   # OMP thread counts to test
    out = []
    for N in sizes:
        for (bm,bn,bk) in blocks:
            # threadpool baseline (workers = cpu_count)
            elapsed, err = run_once(N, bm, bn, bk, mode="threadpool")
            out.append((N, f"{bm}x{bn}x{bk}", "threadpool", "-", elapsed, err))
            for t in threads:
                elapsed, err = run_once(N, bm, bn, bk, mode="openmp", omp_threads=t)
                out.append((N, f"{bm}x{bn}x{bk}", "openmp", t, elapsed, err))
            # flush results to stdout progressively
            print("N,block,mode,threads,elapsed_s,max_err")
            for row in out:
                print(",".join(map(str,row)))
            print("-"*40)
    # optional: write CSV
    with open("bench_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N","block","mode","threads","elapsed_s","max_err"])
        writer.writerows(out)
    print("Wrote bench_results.csv")

if __name__ == "__main__":
    main()

# autotune_blocks.py
import time, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from myext import gemm

def time_config(N, blockM, blockN, blockK, mode="openmp", threads=8):
    import os
    if mode == "openmp":
        os.environ.pop("MYEXT_FORCE_THREADPOOL", None)
        os.environ["OMP_NUM_THREADS"] = str(threads)
    else:
        os.environ["MYEXT_FORCE_THREADPOOL"] = "1"
    # warmup
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))
    gemm(A[:64,:64], B[:64,:64], C[:64,:64], blockM=blockM, blockN=blockN, blockK=blockK)
    t0 = time.time()
    gemm(A, B, C, blockM=blockM, blockN=blockN, blockK=blockK)
    return time.time() - t0

def main():
    N = 1024
    thread_counts = [1,2,4,8]
    candidates = [
        (64,64,64),(32,32,32),(128,128,64),
        (64,64,32),(64,32,32),(32,64,32),
        (128,64,32)
    ]
    best = None
    for (bm,bn,bk) in candidates:
        print("Testing blocks:", bm,bn,bk)
        t_open = time_config(N,bm,bn,bk,mode="openmp",threads=8)
        t_thread = time_config(N,bm,bn,bk,mode="threadpool")
        print(f"openmp={t_open:.4f}s  threadpool={t_thread:.4f}s")
        if best is None or t_open < best[0]:
            best = (t_open, "openmp", bm,bn,bk)
    print("BEST so far:", best)

if __name__ == '__main__':
    main()

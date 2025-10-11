# bench_repeat.py
import time, numpy as np, os
from myext import gemm

def run_once(N, blocks, force_threadpool=False):
    bm,bn,bk = blocks
    if force_threadpool:
        os.environ["MYEXT_FORCE_THREADPOOL"]="1"
    else:
        os.environ.pop("MYEXT_FORCE_THREADPOOL", None)
    A = np.random.rand(N,N).astype(np.float64)
    B = np.random.rand(N,N).astype(np.float64)
    C = np.zeros((N,N), dtype=np.float64)
    t0 = time.time()
    gemm(A,B,C, blockM=bm, blockN=bn, blockK=bk)
    return time.time()-t0

def repeat(N, blocks, mode_name, repeats=5, force_threadpool=False):
    times = [run_once(N, blocks, force_threadpool) for _ in range(repeats)]
    avg = sum(times)/len(times)
    gflops = (2.0 * N**3) / (avg * 1e9)
    print(f"{mode_name}: avg {avg:.4f}s over {repeats} runs — {gflops:.2f} GFLOPS — times: {times}")

if __name__ == "__main__":
    N = 1024
    blocks = (64,64,32)
    print("Using blocks:", blocks)
    repeat(N, blocks, "OpenMP", repeats=5, force_threadpool=False)
    repeat(N, blocks, "Threadpool", repeats=5, force_threadpool=True)

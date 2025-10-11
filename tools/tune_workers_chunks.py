# tools/tune_workers_chunks.py
import time
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer.parallel_pyworkers_v3 import mul_vectors_parallel_py
import itertools
import os

def benchmark(a,b, workers, chunksize, sample_size=20000):
    # run a small sample to estimate per-item cost
    a_s, b_s = a[:sample_size], b[:sample_size]
    t0 = time.perf_counter()
    mul_vectors_parallel_py(a_s, b_s, chunksize=chunksize, max_workers=workers)
    return time.perf_counter() - t0

def full_run(a,b, workers, chunksize):
    t0 = time.perf_counter()
    out = mul_vectors_parallel_py(a, b, chunksize=chunksize, max_workers=workers)
    return time.perf_counter() - t0, len(out)

def main():
    N = 1_000_000
    print("Building inputs N=", N)
    a = [FastDecimal.from_str('123456789.123456789')] * N
    b = [FastDecimal.from_str('987654321.987654321')] * N

    cpu = os.cpu_count() or 1

    # ranges to search (tweak as you like)
    worker_choices = list(range(max(1, cpu//2), cpu+1))  # e.g. 6..12
    chunks_choices = [20000, 25000, 31250, 35000, 40000, 45000, 50000]

    print("Candidates workers:", worker_choices)
    print("Candidates chunksize:", chunks_choices)

    best = None
    best_time = float('inf')

    for w, c in itertools.product(worker_choices, chunks_choices):
        try:
            t = benchmark(a,b, w, c, sample_size=20000)
            est = t  # sample cost for sample_size
            print(f"cand w={w:2d} c={c:6d} sample_time={t:.4f}s")
            if t < best_time:
                best_time = t
                best = (w, c)
        except Exception as e:
            print("error", w, c, e)

    print("BEST (sample) ->", best, "sample_time=", best_time)

    # verify on full run
    if best:
        w,c = best
        full_t, out_len = full_run(a,b,w,c)
        print(f"FULL run workers={w} chunksize={c} -> elapsed {full_t:.3f}s len {out_len}")

if __name__ == "__main__":
    main()

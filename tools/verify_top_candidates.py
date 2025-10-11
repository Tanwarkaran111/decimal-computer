# tools/verify_top_candidates.py
import time
import itertools
import os
import heapq
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer.parallel_pyworkers_v3 import mul_vectors_parallel_py

def sample_benchmark(a,b, workers, chunksize, sample_size=20000):
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
    worker_choices = list(range(max(1, cpu//2), cpu+1))
    chunks_choices = [20000, 25000, 31250, 35000, 40000, 45000, 50000]

    print("Candidates workers:", worker_choices)
    print("Candidates chunksize:", chunks_choices)

    sample_size = 20000
    records = []
    for w, c in itertools.product(worker_choices, chunks_choices):
        try:
            t = sample_benchmark(a,b, w, c, sample_size=sample_size)
            records.append((t, w, c))
            print(f"cand w={w:2d} c={c:6d} sample_time={t:.4f}s")
        except Exception as e:
            print("error", w, c, e)

    # pick top K by sample time
    K = 6
    topk = heapq.nsmallest(K, records, key=lambda x: x[0])
    print("\nTop K candidates (sample):")
    for idx, (t,w,c) in enumerate(topk,1):
        print(f"{idx}. w={w} c={c} sample_time={t:.4f}s")

    print("\nVerifying top K on full run:")
    results = []
    for t_sample, w, c in topk:
        full_t, out_len = full_run(a,b,w,c)
        results.append((full_t, w, c))
        print(f"FULL w={w} c={c} -> elapsed {full_t:.3f}s len {out_len}")

    results.sort()
    best = results[0]
    print("\nBEST full-run -> workers={}, chunksize={}, elapsed {:.3f}s".format(best[1], best[2], best[0]))

if __name__ == "__main__":
    main()

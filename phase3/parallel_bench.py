# phase3/parallel_bench.py
"""
Parallelize benchmark configurations across processes.
Each worker imports phase2 algorithms inside the worker to avoid pickling callables.
"""
import multiprocessing as mp, csv, time, os, random
from typing import Tuple

def gen_matrix(n,m,digits,rng):
    maxv = 10**digits - 1
    return [[rng.randint(0,maxv) for _ in range(m)] for __ in range(n)]

def worker(task):
    # task: (algo_name, module_candidates, n, digits, trial)
    algo_name, n, digits, trial = task
    # import discovery inside worker to avoid pickling issues
    try:
        import importlib
        from phase3.benchmarks import _find_algo_functions
        funcs = _find_algo_functions()
        func = funcs.get(algo_name)
        if func is None:
            return (algo_name, n, digits, trial, None, "missing")
    except Exception as e:
        return (algo_name, n, digits, trial, None, f"import_err:{e}")
    rng = random.Random(12345 + n + digits + trial)
    A = gen_matrix(n,n,digits,rng)
    B = gen_matrix(n,n,digits,rng)
    try:
        # warmup
        _ = func(A, B)
        t0 = time.perf_counter()
        out = func(A, B)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        return (algo_name, n, digits, trial, elapsed, "ok")
    except Exception as e:
        return (algo_name, n, digits, trial, None, f"err:{e}")

def main(sizes, digits_list, algos, trials, processes):
    tasks = []
    for n in sizes:
        for digits in digits_list:
            for trial in range(trials):
                for algo in algos:
                    tasks.append((algo, n, digits, trial))
    out_raw = "phase3_benchmarks_parallel_raw.csv"
    with mp.Pool(processes) as pool, open(out_raw, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo","n","digits","trial","elapsed_seconds","status"])
        for res in pool.imap_unordered(worker, tasks):
            algo,n,digits,trial,elapsed,status = res
            writer.writerow([algo,n,digits,trial,("" if elapsed is None else f"{elapsed:.6f}"),status])
            f.flush()
    print("Done ->", out_raw)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[64,128,256])
    p.add_argument("--digits", nargs="+", type=int, default=[1,2])
    p.add_argument("--algos", nargs="+", default=["schoolbook","karatsuba","strassen","decimal_naive"])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--procs", type=int, default=mp.cpu_count())
    args = p.parse_args()
    main(args.sizes, args.digits, args.algos, args.trials, args.procs)

# phase3/benchmarks_validate.py
"""
Run benchmarks like benchmarks.py but verify correctness by comparing each
algorithm's output vs the canonical schoolbook implementation.
Produces:
 - phase3_benchmarks_validated_raw.csv (columns: algo,size,digits,trial,elapsed,matched)
 - phase3_benchmarks_validated_aggregated.csv (mean/stdev + mismatch count)
"""
import csv, time, statistics, os, random
from typing import List
# reuse discovery and gen_matrix if available
try:
    from phase3.benchmarks import _find_algo_functions, gen_matrix  # if you have phase3/benchmarks.py as module
except Exception:
    # fallback: import the functions from the original script path if present
    # try phase3_benchmarks (older name)
    try:
        from phase3_benchmarks import _find_algo_functions, gen_matrix
    except Exception:
        raise RuntimeError("Cannot import helper functions from existing benchmarks. Ensure phase3/benchmarks.py exists.")

RAW = "phase3_benchmarks_validated_raw.csv"
AGG = "phase3_benchmarks_validated_aggregated.csv"
SEED = 12345

def schoolbook_ref(A: List[List[int]], B: List[List[int]]):
    n = len(A); p = len(B); m = len(B[0])
    C = [[0]*m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik = A[i][k]
            if aik == 0: continue
            brow = B[k]
            rowi = C[i]
            for j in range(m):
                rowi[j] += aik * brow[j]
    return C

def run(sizes, digits_list, algos, trials):
    rng = random.Random(SEED)
    funcs_map = _find_algo_functions()
    # resolve requested algos
    resolved = []
    for a in algos:
        if a in funcs_map:
            resolved.append((a, funcs_map[a]))
        else:
            print(f"[warn] algo {a} not found; skipping")
    if not resolved:
        raise RuntimeError("No algos resolved.")
    # Ensure we have schoolbook ref callable
    ref_callable = funcs_map.get("schoolbook_gemm", schoolbook_ref) if funcs_map else schoolbook_ref

    with open(RAW, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","n","m","p","digits","trial","elapsed_seconds","matched"])
        records = []
        for n in sizes:
            m = p = n
            for digits in digits_list:
                for trial in range(trials):
                    A = gen_matrix(n, p, digits, rng)
                    B = gen_matrix(p, m, digits, rng)
                    # compute canonical reference
                    try:
                        ref = ref_callable(A, B)
                    except Exception as e:
                        print("[error] reference schoolbook raised:", e)
                        ref = None
                    for name, func in resolved:
                        # warmup
                        try:
                            _ = func(A, B)
                        except Exception:
                            print(f"[warn] warmup failed for {name} at n={n} digits={digits}; skipping")
                            continue
                        t0 = time.perf_counter()
                        out = func(A, B)
                        t1 = time.perf_counter()
                        elapsed = t1 - t0
                        matched = (out == ref) if ref is not None else False
                        w.writerow([name, n, m, p, digits, trial, f"{elapsed:.6f}", int(matched)])
                        f.flush()
                        records.append((name,n, digits, elapsed, matched))
    # aggregate
    groups = {}
    mismatches = {}
    for name,n,digits,elapsed,matched in records:
        key = (name,n,digits)
        groups.setdefault(key, []).append(elapsed)
        mismatches.setdefault(key, []).append(0 if matched else 1)
    with open(AGG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","n","digits","trials","mean_seconds","stdev_seconds","mismatch_count"])
        for key, times in sorted(groups.items()):
            name, n, digits = key
            mean_s = statistics.mean(times)
            stdev = statistics.stdev(times) if len(times) > 1 else 0.0
            mismatch_count = sum(mismatches[key])
            w.writerow([name, n, digits, len(times), f"{mean_s:.6f}", f"{stdev:.6f}", mismatch_count])
    print("Done. Raw:", RAW, "Agg:", AGG)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[2,4,8,16,32,64])
    p.add_argument("--digits", nargs="+", type=int, default=[1,2,4])
    p.add_argument("--algos", nargs="+", type=str, default=["schoolbook","karatsuba","strassen","decimal_naive"])
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()
    run(args.sizes, args.digits, args.algos, args.trials)

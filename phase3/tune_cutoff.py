import time
import csv
import os
from phase2.block_strassen import block_strassen
from phase2.parallel_karatsuba import _multiply_block_rows as schoolbook

def timeit(func, A, B, trials=3):
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        func(A, B)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best

def random_matrix(n, digits):
    import random
    rng = random.Random(12345)
    return [[rng.randint(0, 10**digits - 1) for _ in range(n)] for _ in range(n)]

def tune_cutoff(ns=[16, 32, 64, 128], digits_list=[1, 2, 4]):
    results = []
    for digits in digits_list:
        for n in ns:
            A = random_matrix(n, digits)
            B = random_matrix(n, digits)

            # time schoolbook
            t_school = timeit(lambda a,b: schoolbook(a,b), A, B)
            # time strassen
            t_strassen = timeit(lambda a,b: block_strassen(a,b, base_cutoff=n), A, B)

            faster = "schoolbook" if t_school < t_strassen else "strassen"
            cutoff = n if faster == "strassen" else n*2  # heuristic

            results.append({
                "digits": digits,
                "n": n,
                "t_schoolbook": round(t_school, 6),
                "t_strassen": round(t_strassen, 6),
                "preferred": faster,
                "suggested_cutoff": cutoff
            })
            print(f"[digits={digits} n={n}] schoolbook={t_school:.6f}s, strassen={t_strassen:.6f}s -> {faster} (cutoff={cutoff})")

    return results

def save_results(results, out_csv="phase3_cutoff_table.csv"):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote cutoff table -> {out_csv}")

if __name__ == "__main__":
    results = tune_cutoff()
    save_results(results)

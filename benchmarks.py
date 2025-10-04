# benchmarks.py
import random, csv
from decimal_computer.decimal_digit_starter import reset_counters, get_counters
from decimal_computer.decimal_gemm import decimal_gemm_naive

def random_int_with_digits(d):
    if d == 1: return random.randint(0,9)
    return random.randint(10**(d-1), 10**d - 1)

def random_matrix(m, n, digit_len):
    return [[random_int_with_digits(digit_len) for _ in range(n)] for __ in range(m)]

def run_bench(matrix_sizes=(2,4,8,16), digit_lengths=(1,2,3,4), trials=3, out_csv="benchmarks.csv"):
    rows = []
    for d in digit_lengths:
        for size in matrix_sizes:
            m = n = k = size
            for t in range(trials):
                A = random_matrix(m, k, d)
                B = random_matrix(k, n, d)
                C, muls, adds = decimal_gemm_naive(A, B, reset_counters_before=True, return_counters=True)
                rows.append((d, size, t, muls, adds))
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["digit_len","size","trial","muls","adds"])
        w.writerows(rows)
    print("Saved", out_csv)

if __name__ == "__main__":
    run_bench(matrix_sizes=[2,4,8,16], digit_lengths=[1,2,3,4], trials=3)

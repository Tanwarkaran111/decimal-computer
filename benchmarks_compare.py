# benchmarks_compare.py
import random, csv
from Decimal_Computer.decimal_digit_starter import reset_counters, get_counters
from Decimal_Computer.decimal_gemm import decimal_gemm_naive

def random_int_with_digits(d):
    if d == 1: return random.randint(0,9)
    return random.randint(10**(d-1), 10**d - 1)

def random_matrix(m, n, digit_len):
    return [[random_int_with_digits(digit_len) for _ in range(n)] for __ in range(m)]

def run_compare(matrix_sizes=(2,4,8,16), digit_lengths=(1,2,3,4), trials=3, out_csv="benchmarks_compare.csv"):
    rows = []
    for d in digit_lengths:
        for size in matrix_sizes:
            m=n=k=size
            for t in range(trials):
                A = random_matrix(m,k,d)
                B = random_matrix(k,n,d)

                # schoolbook
                C1, muls1, adds1 = decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=True, mul_algo="schoolbook")
                # karatsuba (same inputs)
                C2, muls2, adds2 = decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=True, mul_algo="karatsuba", karatsuba_cutoff=8)

                assert C1 == C2, "Results differ between algos!"

                rows.append((d, size, t, "schoolbook", muls1, adds1))
                rows.append((d, size, t, "karatsuba", muls2, adds2))
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["digit_len","size","trial","algo","muls","adds"])
        w.writerows(rows)
    print("Saved", out_csv)

if __name__ == "__main__":
    run_compare(matrix_sizes=[2,4,8,16], digit_lengths=[1,2,3,4], trials=3)

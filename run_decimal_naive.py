# run_decimal_naive.py
import numpy as np
import time
from numba_decimal import int_to_digits_numba, digits_to_int_numba, naive_digit_mul_numba
from pathlib import Path
import csv

OUT = Path("decimal_naive_results.csv")

def scalar_mul_decimal(a_digits, b_digits):
    # returns digits array for product (maybe large)
    return naive_digit_mul_numba(a_digits, b_digits)

def matrix_naive_decimal_gemm(A_int, B_int, ndigits):
    n = A_int.shape[0]
    C_int = np.zeros((n, n), np.int64)
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                ad = int_to_digits_numba(int(A_int[i,k]), ndigits)
                bd = int_to_digits_numba(int(B_int[k,j]), ndigits)
                prod_digits = scalar_mul_decimal(ad, bd)
                # convert digits->int (careful with overflow)
                s += digits_to_int_numba(prod_digits)
            C_int[i,j] = s
    return C_int

def run(n=16, ndigits=8):
    # for testing: use small n (like 8 or 16)
    A = np.random.randint(0, 10**(ndigits), size=(n,n), dtype=np.int64)
    B = np.random.randint(0, 10**(ndigits), size=(n,n), dtype=np.int64)
    t0 = time.perf_counter()
    C = matrix_naive_decimal_gemm(A, B, ndigits)
    t1 = time.perf_counter()
    print("Result sample:", C[0,0])
    return t1 - t0

if __name__ == "__main__":
    sizes = [8, 16]      # keep small until code is optimized
    ndigits = 8
    rows = []
    for n in sizes:
        t = run(n=n, ndigits=ndigits)
        print(f"n={n}, ndigits={ndigits}, time={t:.4f}")
        rows.append({"n": n, "ndigits": ndigits, "time": t})
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n","ndigits","time"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", OUT)

# example_large_gemm.py
import time, random
from decimal_computer.decimal_gemm import decimal_gemm_naive

def large_random(d):
    # d decimal digits, ensures leading digit nonzero
    return random.randint(10**(d-1), 10**d - 1)

def make_matrix(n, d):
    return [[large_random(d) for _ in range(n)] for __ in range(n)]

def main():
    random.seed(1)
    n = 4                # matrix size (keep small for speed)
    digit_len = 32       # large digits to force Karatsuba
    A = make_matrix(n, digit_len)
    B = make_matrix(n, digit_len)

    # auto
    t0 = time.perf_counter()
    C_auto, muls_auto, adds_auto = decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=True, mul_algo="auto")
    t1 = time.perf_counter()
    print("AUTO: chosen, muls:", muls_auto, "adds:", adds_auto, "time:", t1-t0)

    # force schoolbook
    t0 = time.perf_counter()
    C_s, muls_s, adds_s = decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=True, mul_algo="schoolbook")
    t1 = time.perf_counter()
    print("SCHOOLBOOK: muls:", muls_s, "adds:", adds_s, "time:", t1-t0)

    # force karatsuba (explicit cutoff)
    t0 = time.perf_counter()
    C_k, muls_k, adds_k = decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=True, mul_algo="karatsuba", karatsuba_cutoff=16)
    t1 = time.perf_counter()
    print("KARATSUBA: muls:", muls_k, "adds:", adds_k, "time:", t1-t0)

    assert C_auto == C_s == C_k
    print("OK: results match.")

if __name__ == "__main__":
    main()

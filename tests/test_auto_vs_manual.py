# tests/test_auto_vs_manual.py
from decimal_computer.decimal_digit_starter import int_to_digits
from decimal_computer.decimal_gemm import decimal_gemm_naive

A = [[12, 3], [4, 5]]
B = [[2, 1], [10, 2]]

def test_auto_equals_karatsuba():
    # auto should produce same result as forcing karatsuba
    C_auto = decimal_gemm_naive(A, B, reset_counters_before=True, return_counters=False, mul_algo="auto")
    C_k = decimal_gemm_naive(A, B, reset_counters_before=True, return_counters=False, mul_algo="karatsuba", karatsuba_cutoff=16)
    assert C_auto == C_k

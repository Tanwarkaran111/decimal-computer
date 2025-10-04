# tests/test_adaptive.py
from decimal_computer.decimal_digit_starter import int_to_digits, digits_to_int
from decimal_computer.decimal_gemm import decimal_gemm_naive
from decimal_computer.adaptive import adaptive_gemm

A = [[12, 3], [4, 5]]
B = [[2, 1], [10, 2]]

def test_adaptive_matches_manual():
    # run adaptive path
    C_adaptive, muls_a, adds_a, _ = adaptive_gemm(A, B)

    # run schoolbook explicitly (use project's standard kwargs)
    C_school, muls_s, adds_s = decimal_gemm_naive(
        A, B, reset_counters_before=True, return_counters=True, mul_algo="schoolbook"
    )

    # run karatsuba explicitly
    C_karat, muls_k, adds_k = decimal_gemm_naive(
        A, B, reset_counters_before=True, return_counters=True, mul_algo="karatsuba"
    )

    assert C_adaptive == C_school == C_karat

# example_run_gemm.py
# Quick example showing how to call the adaptive GEMM (auto selection).
import time
from decimal_computer.decimal_gemm import decimal_gemm_naive

def main():
    # Example matrices (small). Change numbers to see algorithm choice.
    A = [[12, 3], [4, 5]]
    B = [[2, 1], [10, 2]]

    # 1) Auto: choose algorithm based on input (and recommended cutoff file)
    t0 = time.perf_counter()
    C_auto, muls_auto, adds_auto = decimal_gemm_naive(
        A, B,
        reset_counters_before=True,
        return_counters=True,
        mul_algo="auto"         # auto selects schoolbook or karatsuba
    )
    t1 = time.perf_counter()
    print("AUTO selected result C =", C_auto)
    print("  digit muls:", muls_auto, "digit adds:", adds_auto, "time (s):", t1 - t0)

    # 2) Force schoolbook
    t0 = time.perf_counter()
    C_s, muls_s, adds_s = decimal_gemm_naive(A, B, reset_counters_before=True, return_counters=True, mul_algo="schoolbook")
    t1 = time.perf_counter()
    print("\nSCHOOLBOOK result C =", C_s)
    print("  digit muls:", muls_s, "digit adds:", adds_s, "time (s):", t1 - t0)

    # 3) Force karatsuba (optionally override cutoff if you want)
    t0 = time.perf_counter()
    C_k, muls_k, adds_k = decimal_gemm_naive(A, B, reset_counters_before=True, return_counters=True, mul_algo="karatsuba", karatsuba_cutoff=16)
    t1 = time.perf_counter()
    print("\nKARATSUBA result C =", C_k)
    print("  digit muls:", muls_k, "digit adds:", adds_k, "time (s):", t1 - t0)

    # Confirm results are identical
    assert C_auto == C_s == C_k, "Mismatch between algorithms!"
    print("\nAll algorithms returned the same result (OK).")

if __name__ == "__main__":
    main()

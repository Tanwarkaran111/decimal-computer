from decimal_computer.decimal_gemm import decimal_gemm_naive

A = [[[1,2,3],[4]], [[5],[6]]]
B = [[[1],[2]], [[3],[4]]]

print(decimal_gemm_naive(A, B, mul_algo="schoolbook", return_counters=True))

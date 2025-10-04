from decimal_computer.decimal_gemm import decimal_gemm_naive

# A = [[123, 4], [5, 6]]
A = [[[1,2,3],[4]], [[5],[6]]]
# B = [[1, 2], [3, 4]]
B = [[[1],[2]], [[3],[4]]]

result = decimal_gemm_naive(A, B, mul_algo="schoolbook", return_counters=True)
print("Result:", result)

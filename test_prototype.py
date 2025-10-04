from decimal_computer import decimal_gemm_naive, int_to_digits, digits_to_int

# Example input matrices (2×2)
A = [[12, 34], [56, 78]]
B = [[87, 65], [43, 21]]

# Run auto GEMM
C = decimal_gemm_naive(A, B, mul_algo="auto", return_counters=True)

print("Result:", C)

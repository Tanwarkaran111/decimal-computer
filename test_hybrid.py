from hybrid_gemm import hybrid_gemm

# Small test (2x2)
A = [[1,2],[3,4]]
B = [[5,6],[7,8]]
print("Result (n=2, d=2):", hybrid_gemm(A, B, digits=2, size=2))

# Larger test (16x16)
A = [[i + j for j in range(16)] for i in range(16)]
B = [[(i * j) % 10 for j in range(16)] for i in range(16)]
res = hybrid_gemm(A, B, digits=8, size=16)

# Print only top-left 2x2 block of result
print("Result (n=16, d=8) top-left 2x2 block:",
      [row[:2] for row in res[:2]])

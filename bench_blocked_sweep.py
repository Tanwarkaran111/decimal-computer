import time, random
from phase2.parallel_karatsuba import _multiply_block_rows as blocked

n = 256
A = [[random.randint(0,9) for _ in range(n)] for __ in range(n)]
B = [[random.randint(0,9) for _ in range(n)] for __ in range(n)]

print("Benchmarking blocked schoolbook with variable block sizes...")
for b in (8, 16, 32, 48, 64):
    t0 = time.perf_counter()
    blocked(A, B, block_size=b)
    print(f"block {b}: {time.perf_counter() - t0:.4f} s")

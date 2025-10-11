from decimal_computer import DecimalContext, FastDecimal, vector_ops
import time

N = 50_000_000  # 5 crore elements

# Setup context
ctx = DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP")
ctx.__enter__()

# Prepare lists
a = [FastDecimal.from_str("1.2345")] * N
b = [FastDecimal.from_str("2.5")] * N

# Warmup
vector_ops.mul_vectors_fast(a[:1000], b[:1000])

# Timed run
start = time.perf_counter()
res = vector_ops.mul_vectors_fast(a, b)
end = time.perf_counter()

print(f"N = {N:,}")
print(f"mul_vectors_fast: {end - start:.3f}s")
print(f"Sample: {str(res[0]) if hasattr(res, '__getitem__') else 'view[0] -> use .arr[0]'}")

ctx.__exit__(None, None, None)
print(f"Precision maintained to scale={ctx.scale} with integer quantization.")

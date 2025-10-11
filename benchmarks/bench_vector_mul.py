# benchmarks/bench_vector_mul.py
"""
Improved benchmark: compare vector_ops.mul_vectors vs naive Decimal loop fairly.

- Ensures inputs are distinct instances (no repeated references).
- Ensures naive path quantizes results the same way we expect FastDecimal results to be quantized.
- Use smaller default N for quick runs; increase for heavier profiling.
"""
from decimal import Decimal, getcontext
import time
from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops

# keep smaller for quick iteration; bump to 100_000 for more stable timings
N = 20000

def make_inputs(n):
    # create distinct instances to avoid aliasing effects
    a = [FastDecimal.from_str("1.2345") for _ in range(n)]
    b = [FastDecimal.from_str("2.5") for _ in range(n)]
    return a, b

def naive_decimal_mul_and_quantize(a, b, ctx_scale, ctx_rounding):
    # perform Decimal multiplication and then quantize like our FastDecimal result
    out = []
    quant = Decimal(1).scaleb(-ctx_scale)
    for ai, bi in zip(a, b):
        d = ai.to_decimal() * bi.to_decimal()
        q = d.quantize(quant, rounding=ctx_rounding)
        out.append(q)
    return out

def run():
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        ctx = DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP")
        # build inputs with distinct objects
        a, b = make_inputs(N)

        # vector_ops path (what we optimized)
        t0 = time.perf_counter()
        res_vector = vector_ops.mul_vectors(a, b)
        t1 = time.perf_counter()

        # naive Decimal path (quantize to same context)
        t2 = time.perf_counter()
        res_naive = naive_decimal_mul_and_quantize(a, b, ctx_scale=4, ctx_rounding="ROUND_HALF_UP")
        t3 = time.perf_counter()

    print("N =", N)
    print("vector_ops.mul_vectors: {:.3f}s".format(t1 - t0))
    print("naive Decimal loop     : {:.3f}s".format(t3 - t2))
    # correctness checks: compare first 5 elements
    for i in range(5):
        fd = str(res_vector[i])
        naive_q = format(res_naive[i], 'f').rstrip('0').rstrip('.')
        print(f"i={i}: vector={fd}  naive={naive_q}")
    # quick assert for first element (not raising in production)
    print("sample equality (first):", str(res_vector[0]) == (format(res_naive[0], 'f').rstrip('0').rstrip('.')))

if __name__ == "__main__":
    run()

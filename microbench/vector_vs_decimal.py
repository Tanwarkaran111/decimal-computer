# microbench/vector_vs_decimal.py
"""
Microbenchmark: compare three approaches for elementwise multiplication:
  - vector_ops.mul_vectors (optimized integer-array path)
  - naive Decimal loop (convert each to Decimal, multiply, quantize)
  - naive FastDecimal per-element (call FastDecimal.__mul__ for each pair)

Run:
    python microbench/vector_vs_decimal.py
"""
import time
from decimal import Decimal
import random

from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops

random.seed(42)

def make_inputs(n, s1="1.2345", s2="2.5"):
    # create distinct instances
    a = [FastDecimal.from_str(s1) for _ in range(n)]
    b = [FastDecimal.from_str(s2) for _ in range(n)]
    return a, b

def naive_decimal_mul_and_quantize(a, b, ctx_scale, ctx_rounding):
    quant = Decimal(1).scaleb(-ctx_scale)
    out = []
    for ai, bi in zip(a, b):
        d = ai.to_decimal() * bi.to_decimal()
        q = d.quantize(quant, rounding=ctx_rounding)
        out.append(q)
    return out

def naive_fastdecimal_loop(a, b):
    out = []
    for ai, bi in zip(a, b):
        out.append(ai * bi)
    return out

def run():
    N = 20000
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a, b = make_inputs(N)
        # vector_ops
        t0 = time.perf_counter()
        vres = vector_ops.mul_vectors(a, b)
        t1 = time.perf_counter()

        # naive decimal
        t2 = time.perf_counter()
        dres = naive_decimal_mul_and_quantize(a, b, ctx_scale=4, ctx_rounding="ROUND_HALF_UP")
        t3 = time.perf_counter()

        # naive fastdecimal
        t4 = time.perf_counter()
        fres = naive_fastdecimal_loop(a, b)
        t5 = time.perf_counter()

    print(f"N={N}")
    print("vector_ops.mul_vectors: {:.3f}s".format(t1 - t0))
    print("naive Decimal loop     : {:.3f}s".format(t3 - t2))
    print("naive FastDecimal loop : {:.3f}s".format(t5 - t4))
    # quick correctness checks
    for i in range(3):
        print("sample:", str(vres[i]), format(dres[i], 'f').rstrip('0').rstrip('.'))

if __name__ == '__main__':
    run()

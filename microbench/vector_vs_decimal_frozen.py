# microbench/vector_vs_decimal_frozen.py
"""
Compare vector mul with/without freezing inputs to force integer-backed paths.
Run:
    python microbench/vector_vs_decimal_frozen.py
"""
import time
from decimal import Decimal
from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops

def make_inputs(n, s1="1.2345", s2="2.5", freeze=False):
    a = [FastDecimal.from_str(s1) for _ in range(n)]
    b = [FastDecimal.from_str(s2) for _ in range(n)]
    if freeze:
        a = FastDecimal.freeze_list(a)
        b = FastDecimal.freeze_list(b)
    return a, b

def naive_decimal_mul_and_quantize(a, b, ctx_scale, ctx_rounding):
    quant = Decimal(1).scaleb(-ctx_scale)
    out = []
    for ai, bi in zip(a, b):
        d = ai.to_decimal() * bi.to_decimal()
        q = d.quantize(quant, rounding=ctx_rounding)
        out.append(q)
    return out

def run():
    N = 20000
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        # baseline: inputs preserve orig_decimal
        a1, b1 = make_inputs(N, freeze=False)
        t0 = time.perf_counter()
        v1 = vector_ops.mul_vectors(a1, b1)
        t1 = time.perf_counter()

        # frozen: convert to pure integer-backed first
        a2, b2 = make_inputs(N, freeze=True)
        t2 = time.perf_counter()
        v2 = vector_ops.mul_vectors(a2, b2)
        t3 = time.perf_counter()

        # naive decimal
        t4 = time.perf_counter()
        d = naive_decimal_mul_and_quantize(a1, b1, ctx_scale=4, ctx_rounding="ROUND_HALF_UP")
        t5 = time.perf_counter()

    print(f"N = {N}")
    print("baseline (from_str inputs) : {:.3f}s".format(t1 - t0))
    print("frozen inputs (freeze_list): {:.3f}s".format(t3 - t2))
    print("naive Decimal loop         : {:.3f}s".format(t5 - t4))
    print("sample equality baseline==decimal:", str(v1[0]) == format(d[0], 'f').rstrip('0').rstrip('.'))
    print("sample equality frozen==decimal  :", str(v2[0]) == format(d[0], 'f').rstrip('0').rstrip('.'))

if __name__ == "__main__":
    run()

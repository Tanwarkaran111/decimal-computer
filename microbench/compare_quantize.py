# microbench/compare_quantize.py
"""
Microbenchmark: compare quantize_int performance vs Decimal.quantize for various sizes.

Run:
    python microbench/compare_quantize.py
"""
import time
from decimal import Decimal
import random

from decimal_computer.rounding import quantize_int

random.seed(12345)

def bench_single(orig_scale, target_scale, rounds=10000):
    vals = [random.randint(-10**6, 10**6) for _ in range(rounds)]

    t0 = time.perf_counter()
    for v in vals:
        dec = Decimal(v).scaleb(-orig_scale)
        q = dec.quantize(Decimal(1).scaleb(-target_scale), rounding="ROUND_HALF_EVEN")
        _ = int(q.scaleb(target_scale).to_integral_value(rounding="ROUND_HALF_EVEN"))
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for v in vals:
        _ = quantize_int(v, orig_scale=orig_scale, target_scale=target_scale, rounding="ROUND_HALF_EVEN")
    t3 = time.perf_counter()

    return (t1 - t0), (t3 - t2)

def run():
    cases = [
        (6, 4),
        (4, 2),
        (8, 3),
        (3, 3),
        (2, 6),
    ]
    rounds = 20000
    print(f"rounds={rounds}")
    for orig, targ in cases:
        dec_t, int_t = bench_single(orig, targ, rounds=rounds)
        print(f"orig={orig} target={targ} -> Decimal: {dec_t:.3f}s  quantize_int: {int_t:.3f}s  ratio={dec_t/int_t if int_t>0 else float('inf'):.2f}")

if __name__ == "__main__":
    run()

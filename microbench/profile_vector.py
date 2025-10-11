# microbench/profile_vector.py
"""
Profile vector_ops.mul_vectors hot path with cProfile.

Run:
    python microbench/profile_vector.py

It runs the vector benchmark and prints the top callers by cumulative time.
"""
import cProfile
import pstats
import io
from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops
from decimal import Decimal
import time

def make_inputs(n):
    a = [FastDecimal.from_str("1.2345") for _ in range(n)]
    b = [FastDecimal.from_str("2.5") for _ in range(n)]
    return a, b

def profile(n=20000):
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a, b = make_inputs(n)
        profiler = cProfile.Profile()
        profiler.enable()
        _ = vector_ops.mul_vectors(a, b)
        profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)  # top 30 entries
    print(s.getvalue())

if __name__ == "__main__":
    profile()

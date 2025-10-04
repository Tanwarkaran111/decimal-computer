import random
from phase3.impl_wrappers import WRAPPERS
from phase2.decimal_gemm import decimal_gemm_naive
# tests/test_wrappers.py (top)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# rest of the test file...

def check_wrappers(n=64):
    A = [[random.randint(0,9) for _ in range(n)] for __ in range(n)]
    B = [[random.randint(0,9) for _ in range(n)] for __ in range(n)]
    base = decimal_gemm_naive(A,B)
    bad = []
    for name, fn in WRAPPERS.items():
        r = fn(A,B,cutoff=64)
        if r != base:
            bad.append(name)
    if bad:
        print("MISMATCH:", bad)
        return False
    else:
        print("OK: all wrappers match decimal_naive")
        return True

if __name__ == "__main__":
    check_wrappers()

# tools/profile_bigint_mul.py
# tools/profile_bigint_mul.py (top)
import sys
from pathlib import Path
# ensure repo root is on sys.path so sibling packages import correctly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cProfile, pstats, io
from phase7.bigint import BigInt
import time

def run(d=20000):
    a = int("9" * d)
    b = int("8" * d)
    A = BigInt.from_int(a)
    B = BigInt.from_int(b)
    pr = cProfile.Profile()
    pr.enable()
    _ = A * B
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(30)
    print(s.getvalue())

if __name__ == "__main__":
    run()

# phase2/bench_compare.py
import time
import logging

# silence noisy module-level logs from hybrid_core (optional)
logging.getLogger("hybrid_core").setLevel(logging.WARNING)

try:
    import numpy as np
    have_numpy = True
    rng = np.random.default_rng()
except Exception:
    have_numpy = False
    rng = None

from phase2.hybrid_core import hybrid_gemm


def make_random_matrix(n: int, digits: int):
    high = 10 ** digits
    if have_numpy and rng is not None:
        try:
            # use Generator.integers with explicit int64 dtype so large highs work
            arr = rng.integers(0, high, size=(n, n), dtype="int64")
            return arr.tolist()        # convert to nested python lists
        except Exception:
            pass
    # fallback: pure python random (always safe)
    import random
    return [[random.randrange(0, high) for _ in range(n)] for _ in range(n)]


def bench_once(n, digits, trials=3):
    A = make_random_matrix(n, digits)
    B = make_random_matrix(n, digits)

    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = hybrid_gemm(A, B)   # call with matrices (hybrid_gemm expects A,B)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def main():
    for digits in [4, 8, 16]:
        print(f"\n=== Digits={digits} ===")
        for n in [32, 64, 128]:
            try:
                t = bench_once(n, digits, trials=2)
                print(f"n={n}, d={digits} -> {t:.6f}s")
            except Exception as e:
                print(f"n={n}, d={digits} -> ERROR: {e}")


if __name__ == "__main__":
    main()

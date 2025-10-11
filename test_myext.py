# test_myext.py -- run from repo root with: python .\test_myext.py
import sys
import time
import traceback
import numpy as np
import os

# make local src importable
sys.path.insert(0, "src")

def main():
    print("PYTHON:", sys.executable)
    print("CACHED TAG:", getattr(sys.implementation, "cache_tag", None))
    print("working dir:", os.getcwd())
    print()

    try:
        import myext
        import myext.helpers as helpers
    except Exception:
        print("FAILED to import myext or helpers:")
        traceback.print_exc()
        return 2

    print("myext module file:", getattr(myext, "__file__", None))
    members = [m for m in dir(myext) if not m.startswith("_")]
    print("myext members:", members)

    _g = getattr(helpers, "_g", None)
    print("helpers._g ->", type(_g), getattr(_g, "__file__", None))

    if _g is None:
        print("NOTE: compiled backend (_g) is None -> using Python fallback (threadpool).")
    else:
        print("native backend functions present:", [a for a in dir(_g) if a.startswith("gemm")])
        print("has gemm_openmp:", hasattr(_g, "gemm_openmp"))
        print("has gemm_block :", hasattr(_g, "gemm_block"))

    # quick correctness check
    try:
        N = 128
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        t0 = time.perf_counter()
        C = myext.gemm(A, B)
        t1 = time.perf_counter()
        print("gemm returned shape:", getattr(C, "shape", None))
        print(f"time (myext.gemm) : {t1 - t0:.6f} s")
        # sanity-check correctness against numpy
        diff = np.max(np.abs(C - A.dot(B)))
        print("max error vs numpy.dot:", diff)
    except Exception:
        print("ERROR while running myext.gemm():")
        traceback.print_exc()
        return 3

    # if native backend exists, try calling its function directly (best-effort)
    if _g is not None:
        try:
            if hasattr(_g, "gemm_openmp"):
                C2 = np.zeros_like(C)
                t0 = time.perf_counter()
                _g.gemm_openmp(A, B, C2, 64, 64, 32)
                t1 = time.perf_counter()
                print("native gemm_openmp time:", t1 - t0)
                print("native max error vs numpy.dot:", np.max(np.abs(C2 - A.dot(B))))
            elif hasattr(_g, "gemm_block"):
                print("native has gemm_block (block API available).")
            else:
                print("native backend loaded but no expected entrypoints found.")
        except Exception:
            print("Calling native functions failed (traceback):")
            traceback.print_exc()

    print("\nDone.")
    return 0

if __name__ == "__main__":
    rc = main()
    # exit non-zero on failures when run from shell
    sys.exit(rc)

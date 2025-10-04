import cProfile, pstats
import numpy as np
from phase2.hybrid_core import hybrid_gemm   # ✅ fixed import

def profile_case(n=64, digits=8):
    A = np.random.randint(0, 10**digits, size=(n, n)).tolist()
    B = np.random.randint(0, 10**digits, size=(n, n)).tolist()
    hybrid_gemm(A, B, digits=digits, size=n)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_case()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)  # show top 20 time-consuming functions

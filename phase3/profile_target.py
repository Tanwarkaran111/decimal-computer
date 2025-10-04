# phase3/profile_target.py
"""
Profile a single run of a chosen algorithm using cProfile.
"""
import cProfile, pstats, os, argparse, importlib, time

def main(algo, n, digits, out):
    # Discover function using benchmarks discovery
    from phase3.benchmarks import _find_algo_functions
    funcs = _find_algo_functions()
    func = funcs.get(algo)
    if func is None:
        raise RuntimeError("Algo not found: "+algo)
    # create test matrices
    import random
    rng = random.Random(12345)
    A = [[rng.randint(0,10**digits-1) for j in range(n)] for i in range(n)]
    B = [[rng.randint(0,10**digits-1) for j in range(n)] for i in range(n)]
    profiler = cProfile.Profile()
    profiler.enable()
    func(A,B)
    profiler.disable()
    profiler.dump_stats(out)
    print("Wrote profile:", out)
    # optionally print top stats
    ps = pstats.Stats(out)
    ps.strip_dirs().sort_stats("cumtime").print_stats(40)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="strassen")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--digits", type=int, default=1)
    p.add_argument("--out", default="profile.prof")
    args = p.parse_args()
    main(args.algo, args.n, args.digits, args.out)

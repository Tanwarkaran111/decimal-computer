# tests/test_integration.py
import math
import random
from phase3.benchmarks import _find_algo_functions

RNG = random.Random(12345)

def small_random_int(digits=5):
    return RNG.randint(10**(digits-1), 10**digits - 1)

def test_all_algorithms_agree_on_small_examples():
    funcs = _find_algo_functions()
    # choose a few random small ints
    cases = [(small_random_int(3), small_random_int(3)) for _ in range(8)]
    # include deterministic corner cases
    cases += [(0, 0), (0, 12345), (99999, 1), (12345, 67890)]
    results = {}
    for name, fn in funcs.items():
        # we only care about numeric multiply-style callables; wrap if necessary
        reslist = []
        for a, b in cases:
            try:
                r = fn(a, b)
            except TypeError:
                # try common wrapper names: allow functions that expect ints
                r = fn(int(a), int(b))
            reslist.append(int(r))
        results[name] = reslist

    # assert that all present algorithms produce same outputs as 'fft' if available,
    # otherwise pick any algorithm as reference
    ref = results.get("fft") or next(iter(results.values()))
    for name, res in results.items():
        assert res == ref, f"algo {name} disagrees with reference"

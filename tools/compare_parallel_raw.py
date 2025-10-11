# tools/compare_parallel_raw.py
"""
Compare mul_vectors_parallel vs mul_vectors_raw on identical inputs and print diagnostics.
Run from project root: python tools/compare_parallel_raw.py
"""
from decimal_computer import DecimalContext, FastDecimal
from decimal_computer.context import get_context
from decimal_computer.vector_ops import mul_vectors_raw
from decimal_computer.parallel import mul_vectors_parallel
from decimal import Decimal as D
from array import array
import random

def make_sample(n=20, seed=42):
    rnd = random.Random(seed)
    a = []
    b = []
    for i in range(n):
        int_part = rnd.randint(-10000, 10000)
        frac_len = rnd.randint(0, 4)
        frac_val = rnd.randint(0, 10**frac_len - 1) if frac_len > 0 else 0
        if frac_len == 0:
            s1 = str(int_part)
        else:
            sign = "-" if int_part < 0 else ""
            s1 = f"{sign}{abs(int_part)}.{str(frac_val).zfill(frac_len)}"
        a.append(FastDecimal.from_str(s1))

        if rnd.random() < 0.6:
            s2 = str(rnd.randint(-50, 50))
        else:
            fd2 = rnd.randint(0, 3)
            frac2 = rnd.randint(0, 10**fd2 - 1) if fd2 > 0 else 0
            s2 = f"{rnd.randint(-50,50)}.{str(frac2).zfill(fd2)}"
        b.append(FastDecimal.from_str(s2))
    return a, b

def to_list_of_ints(arr_like):
    # accept array('q') or FastDecimalView-like with .arr or list
    if isinstance(arr_like, array):
        return list(arr_like)
    try:
        return list(arr_like.arr)
    except Exception:
        # fallback: try list()
        return list(arr_like)

def run_check(n=50):
    with DecimalContext(precision=50, scale=6, rounding="ROUND_HALF_EVEN", mode="auto"):
        a, b = make_sample(n=n, seed=2025)
        ctx = get_context()
        print("Main process ctx.scale:", ctx.scale, "rounding:", ctx.rounding)

        print("\nCalling mul_vectors_raw...")
        raw = mul_vectors_raw(a, b)
        raw_ints = to_list_of_ints(raw)
        print("mul_vectors_raw returned type:", type(raw), "len:", len(raw_ints))
        print("first 10 raw ints:", raw_ints[:10])
        print("first 10 raw values:", [D(x).scaleb(-ctx.scale) for x in raw_ints[:10]])

        print("\nCalling mul_vectors_parallel...")
        par = mul_vectors_parallel(a, b, chunksize=1000)
        try:
            par_arr = par.arr
            par_ints = list(par_arr)
        except Exception:
            # maybe it's array('q')
            if isinstance(par, array):
                par_ints = list(par)
            else:
                try:
                    par_ints = list(par)
                except Exception:
                    print("Unexpected return type from mul_vectors_parallel:", type(par))
                    raise

        print("mul_vectors_parallel returned type:", type(par), "len:", len(par_ints))
        print("first 10 parallel ints:", par_ints[:10])
        print("first 10 parallel values:", [D(x).scaleb(-ctx.scale) for x in par_ints[:10]])

        # Compare itemwise
        mismatches = []
        for i, (r, p) in enumerate(zip(raw_ints, par_ints)):
            if r != p:
                mismatches.append((i, r, p))
        if not mismatches:
            print("\nOK: raw and parallel ints match elementwise for first", len(raw_ints), "elements.")
        else:
            print("\nFound mismatches between raw and parallel outputs (showing up to 20):")
            for i, r, p in mismatches[:20]:
                print(f" idx {i}: raw_int={r} ({D(r).scaleb(-ctx.scale)}), par_int={p} ({D(p).scaleb(-ctx.scale)})")
            print(f"... total mismatches: {len(mismatches)}")
        return raw_ints, par_ints, mismatches

if __name__ == "__main__":
    run_check(n=200)

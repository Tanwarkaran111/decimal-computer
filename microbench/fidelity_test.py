# microbench/fidelity_test.py  (updated with extra diagnostics)
from decimal_computer import DecimalContext, FastDecimal
from decimal_computer.context import get_context
from decimal import Decimal as D, localcontext
from decimal_computer.parallel_pyworkers import mul_vectors_parallel_py as mul_vectors_parallel
from decimal_computer.vector_ops import mul_vectors_raw  # raw path (inspects C output)
from decimal_computer.rounding import quantize_int
import random
from array import array
import sys

def debug_case(a_fd, b_fd):
    """
    Reproduce and print a single-case debug comparison for two FastDecimal inputs.
    Returns a tuple (c_int, py_int, expected_dec).
    """
    ctx = get_context()
    target_scale = ctx.scale
    rounding = ctx.rounding

    # call raw vector multiply to get C-kernel output array('q')
    arr = mul_vectors_raw([a_fd], [b_fd])
    if isinstance(arr, array):
        c_int = int(arr[0])
    else:
        # if mul_vectors_raw returned a view-like object
        try:
            c_int = int(arr[0])
        except Exception:
            raise RuntimeError("unexpected raw return type: %r" % type(arr))

    # python integer path
    prod = a_fd.int_value * b_fd.int_value
    py_int = quantize_int(prod, orig_scale=(a_fd.scale + b_fd.scale), target_scale=target_scale, rounding=rounding)

    # high-precision Decimal expected
    with localcontext() as lc:
        lc.prec = 120
        quant = D(1).scaleb(-target_scale)
        expected = (D(str(a_fd)) * D(str(b_fd))).quantize(quant, rounding=rounding)

    # print detailed debug
    print("=== Debug case ===")
    print("a FastDecimal:", a_fd, " -> int_value:", a_fd.int_value, " scale:", a_fd.scale)
    print("b FastDecimal:", b_fd, " -> int_value:", b_fd.int_value, " scale:", b_fd.scale)
    print("target_scale:", target_scale, "rounding:", rounding)
    print("raw C kernel int:", c_int, " -> value:", D(c_int).scaleb(-target_scale))
    print("python int path int:", py_int, " -> value:", D(py_int).scaleb(-target_scale))
    print("Decimal expected:", expected)
    print("prod (a_int*b_int) =", prod)
    return c_int, py_int, expected

def compare_chunks(N=200_000, chunksize=100_000, trials=5, max_diag=30):
    mismatches = 0
    with DecimalContext(precision=50, scale=6, rounding="ROUND_HALF_EVEN", mode="auto"):
        for t in range(trials):
            print(f"Trial {t+1}/{trials} building inputs N={N}")
            a = []
            b = []
            for i in range(N):
                int_part = random.randint(-10000, 10000)
                frac_len = random.randint(0, 4)
                frac_val = random.randint(0, 10**frac_len - 1) if frac_len > 0 else 0
                if frac_len == 0:
                    s1 = str(int_part)
                else:
                    sign = "-" if int_part < 0 else ""
                    s1 = f"{sign}{abs(int_part)}.{str(frac_val).zfill(frac_len)}"
                a.append(FastDecimal.from_str(s1))

                if random.random() < 0.6:
                    s2 = str(random.randint(-50, 50))
                else:
                    fd2 = random.randint(0, 3)
                    frac2 = random.randint(0, 10**fd2 - 1) if fd2 > 0 else 0
                    s2 = f"{random.randint(-50,50)}.{str(frac2).zfill(fd2)}"
                b.append(FastDecimal.from_str(s2))

            print("Computing via mul_vectors_parallel...")
            arr = mul_vectors_parallel(a, b, chunksize=chunksize)

            # Accept FastDecimalView-like return (has .arr) or plain array('q')
            if not isinstance(arr, array):
                try:
                    arr = arr.arr  # FastDecimalView case
                except Exception:
                    raise RuntimeError("mul_vectors_parallel returned unexpected type: %r" % type(arr))

            if len(arr) != len(a):
                print("ERROR: result length mismatch!")
                print("len(a) =", len(a), "len(arr) =", len(arr))
                print("First few arr values:", list(arr[:10]))

            limit = min(len(arr), len(a))

            # active context for target scale and rounding
            ctx = get_context()
            target_scale = ctx.scale
            rounding = ctx.rounding

            with localcontext() as lc:
                lc.prec = 110  # high precision for expected
                quant = D(1).scaleb(-target_scale)

                for i in range(limit):
                    # C kernel result
                    c_int = int(arr[i])
                    c_dec = D(c_int).scaleb(-target_scale)

                    # Python integer path: quantize_int(a.int_value * b.int_value, ...)
                    prod = a[i].int_value * b[i].int_value
                    py_int = quantize_int(prod, orig_scale=(a[i].scale + b[i].scale), target_scale=target_scale, rounding=rounding)
                    py_dec = D(py_int).scaleb(-target_scale)

                    # Decimal expected path: high-precision multiply then quantize
                    expected = (D(str(a[i])) * D(str(b[i]))).quantize(quant, rounding=rounding)

                    # Compare using Decimal equality (treats +0/-0 equal)
                    if c_dec != expected:
                        mismatches += 1
                        if mismatches <= max_diag:
                            print("\nMismatch idx=", i)
                            print("  a =", a[i], "b =", b[i])
                            print("  C kernel  ->", c_dec, "(int:", c_int, ")")
                            print("  Python int->", py_dec, "(int:", py_int, ")")
                            print("  Decimal   ->", expected)

                            # Additional diagnostics: show raw int inputs & scales
                            print("  debug inputs -> a.int_value:", a[i].int_value, "a.scale:", a[i].scale,
                                  "b.int_value:", b[i].int_value, "b.scale:", b[i].scale,
                                  "prod(a_int*b_int):", prod)
                            # Quick note whether C kernel matches python int path or not
                            if c_dec == py_dec:
                                print("  NOTE: C kernel == python integer quantize (so Decimal differs).")
                            else:
                                print("  NOTE: C kernel != python integer quantize (C kernel likely wrong).")

                            # Reproduce single-case debug (calls mul_vectors_raw)
                            try:
                                c_int2, py_int2, expected2 = debug_case(a[i], b[i])
                                print("  Reproducer: raw C int:", c_int2, "py_int:", py_int2, "expected:", expected2)
                            except Exception as e:
                                print("  Reproducer failed with:", e)

                if len(arr) != len(a):
                    print("Warning: result length != inputs; compared up to min(len). Continuing to next trial.")

    print("Total mismatches:", mismatches)
    if mismatches:
        # nonzero exit so CI / scripts know it's not passing
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    # small quick run if desired: compare_chunks(N=20000, chunksize=5000, trials=1)
    compare_chunks(N=200000, chunksize=100000, trials=5)

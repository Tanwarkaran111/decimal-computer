# tools/debug_c_quantize.py
"""
Quick diagnostic for _c_quantize <-> Python interface mismatch.

What it does:
 - Prints which _c_quantize module file is imported.
 - Shows the callable signature detected at import-time.
 - Runs a small randomized test (default 2000 pairs) comparing:
     * call via legacy 6-arg signature:  c_mul_and_quantize_from_arrays(a, b, a_scale, b_scale, target_scale, rounding)
     * call via new 5-arg signature:     c_mul_and_quantize_from_arrays(a, b, orig_scale, target_scale, rounding)
     * python integer quantize path using quantize_int
     * Decimal high-precision expected
 - Prints full details for the first few mismatches to help pinpoint scale/arg confusion.
"""

import importlib
import inspect
import os
import random
from array import array
from decimal import Decimal as D, localcontext

# Adjust import path if needed; assume package layout is root/tools/debug_c_quantize.py
# and decimal_computer is a sibling package (project root).
try:
    import decimal_computer._c_quantize as cq
    from decimal_computer.rounding import quantize_int
    from decimal_computer.fastdecimal import FastDecimal
    from decimal_computer.context import DecimalContext, get_context
except Exception as e:
    raise SystemExit(f"Failed to import project modules: {e}")

def module_info():
    print("Imported module:", cq)
    try:
        print("module file:", cq.__file__)
    except Exception:
        print("module file: <unknown>")
    print("callable attributes in module:", [n for n in dir(cq) if n.startswith("c_")])
    # show signature if Python wrapper exists
    try:
        sig = inspect.signature(cq.c_mul_and_quantize_from_arrays)
        print("Detected signature for c_mul_and_quantize_from_arrays:", sig)
    except Exception as e:
        print("Could not get signature:", e)

def build_array_from_fd_list(fd_list):
    """Return array('q') of int_value from list of FastDecimal-like objects."""
    return array("q", [int(fd.int_value) for fd in fd_list])

def run_quick_checks(n=2000, max_print=10):
    mismatches = 0
    printed = 0

    # use a small deterministic seed so results are reproducible
    rnd = random.Random(12345)

    # set same context as your tests: scale=6 rounding half-even
    with DecimalContext(precision=50, scale=6, rounding="ROUND_HALF_EVEN", mode="auto"):
        ctx = get_context()
        target_scale = ctx.scale
        rounding = ctx.rounding

        for i in range(n):
            # build random pair similar to your test generator
            int_part = rnd.randint(-10000, 10000)
            frac_len = rnd.randint(0, 4)
            frac_val = rnd.randint(0, 10**frac_len - 1) if frac_len > 0 else 0
            if frac_len == 0:
                s1 = str(int_part)
            else:
                sign = "-" if int_part < 0 else ""
                s1 = f"{sign}{abs(int_part)}.{str(frac_val).zfill(frac_len)}"
            a_fd = FastDecimal.from_str(s1)

            if rnd.random() < 0.6:
                s2 = str(rnd.randint(-50, 50))
            else:
                fd2 = rnd.randint(0, 3)
                frac2 = rnd.randint(0, 10**fd2 - 1) if fd2 > 0 else 0
                s2 = f"{rnd.randint(-50,50)}.{str(frac2).zfill(fd2)}"
            b_fd = FastDecimal.from_str(s2)

            # arrays
            a_arr = array("q", [a_fd.int_value])
            b_arr = array("q", [b_fd.int_value])

            # Compute Python references
            prod = a_fd.int_value * b_fd.int_value
            py_int = quantize_int(prod, orig_scale=(a_fd.scale + b_fd.scale), target_scale=target_scale, rounding=rounding)
            with localcontext() as lc:
                lc.prec = 120
                quant = D(1).scaleb(-target_scale)
                expected = (D(str(a_fd)) * D(str(b_fd))).quantize(quant, rounding=rounding)

            # Try calling compiled kernel in both legacy and new forms,
            # catching exceptions and printing what signature was actually handled.
            c_results = {}
            # try new 5-arg form
            try:
                # orig_scale = a_scale + b_scale (for single-pair)
                orig_scale = a_fd.scale + b_fd.scale
                out = cq.c_mul_and_quantize_from_arrays(a_arr, b_arr, orig_scale, target_scale, 0 if rounding == "ROUND_HALF_EVEN" else 1)
                # out may be (res, mask) or res only
                if isinstance(out, (tuple, list)):
                    res_arr, mask = out
                else:
                    res_arr = out
                    mask = None
                c_results["5arg"] = (int(res_arr[0]), mask[0] if mask is not None else None)
            except Exception as e:
                c_results["5arg"] = ("exc", str(e))

            # try legacy 6-arg form
            try:
                out2 = cq.c_mul_and_quantize_from_arrays(a_arr, b_arr, a_fd.scale, b_fd.scale, target_scale, 0 if rounding == "ROUND_HALF_EVEN" else 1)
                if isinstance(out2, (tuple, list)):
                    res2_arr, mask2 = out2
                else:
                    res2_arr = out2
                    mask2 = None
                c_results["6arg"] = (int(res2_arr[0]), mask2[0] if mask2 is not None else None)
            except Exception as e:
                c_results["6arg"] = ("exc", str(e))

            # Evaluate mismatches vs expected (Decimal)
            # Choose the most plausible C result if available (prefer 5arg)
            chosen = None
            for k in ("5arg", "6arg"):
                v = c_results.get(k)
                if v is None:
                    continue
                if isinstance(v, tuple) and isinstance(v[0], int):
                    chosen = v[0]
                    chosen_form = k
                    break
            # If none returned a numeric value, skip
            if chosen is None:
                # if both raised, count as mismatch for visibility
                mismatches += 1
                if printed < max_print:
                    print(f"ITER {i}: C calls failed: {c_results}")
                    printed += 1
                continue

            c_dec = D(chosen).scaleb(-target_scale)
            # compare Decimal equality
            if c_dec != expected:
                mismatches += 1
                if printed < max_print:
                    print("\n--- MISMATCH SAMPLE ---")
                    print("iter:", i)
                    print("a_str:", s1, "b_str:", s2)
                    print("a.int:", a_fd.int_value, "a.scale:", a_fd.scale)
                    print("b.int:", b_fd.int_value, "b.scale:", b_fd.scale)
                    print("prod:", prod)
                    print("target_scale:", target_scale, "rounding:", rounding)
                    print("python int quantized:", py_int, "->", D(py_int).scaleb(-target_scale))
                    print("Decimal expected:", expected)
                    print("c_results:", c_results)
                    print("chosen form:", chosen_form, "c_int:", chosen, "->", c_dec)
                    printed += 1

    print("\nSummary: tested", n, "pairs. mismatches:", mismatches)
    return mismatches

if __name__ == "__main__":
    print("=== Module info ===")
    module_info()
    print("\n=== Running quick checks ===")
    mism = run_quick_checks(n=2000, max_print=12)
    if mism:
        print("Some mismatches found. See above output for samples.")
    else:
        print("No mismatches detected in quick run.")

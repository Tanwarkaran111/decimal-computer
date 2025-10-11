# tools/bench_batch_quantize.py
import argparse
import time
from decimal_computer.batch_quantize import batch_quantize as batch_q
from decimal_computer.rounding import quantize_int
from decimal_computer.fastdecimal import FastDecimal

def build_inputs(n):
    a = [FastDecimal.from_str("1.23") for _ in range(n)]
    b = [FastDecimal.from_str("4.56") for _ in range(n)]
    a_ints = [int(x.int_value) for x in a]
    b_ints = [int(x.int_value) for x in b]
    a_sc = [int(x.scale) for x in a]
    b_sc = [int(x.scale) for x in b]
    return a_ints, b_ints, a_sc, b_sc

def run_cython(a_ints, b_ints, a_sc, b_sc, target_scale, rounding):
    t0 = time.perf_counter()
    out = batch_q(a_ints, b_ints, a_sc, b_sc, target_scale, rounding)
    elapsed = time.perf_counter() - t0
    return out, elapsed

def run_python(a_ints, b_ints, a_sc, b_sc, target_scale, rounding):
    out = []
    t0 = time.perf_counter()
    for ai, bi, asc, bsc in zip(a_ints, b_ints, a_sc, b_sc):
        prod = int(ai) * int(bi)
        orig_scale = int(asc) + int(bsc)
        q = quantize_int(prod, orig_scale=orig_scale, target_scale=target_scale, rounding=rounding)
        out.append(int(q))
    elapsed = time.perf_counter() - t0
    return out, elapsed

def main():
    p = argparse.ArgumentParser(description="Benchmark batch_quantize vs python quantize_int")
    p.add_argument("--n", type=int, default=200000, help="number of elements")
    p.add_argument("--target_scale", type=int, default=2, help="target integer scale for quantization")
    p.add_argument("--rounding", type=str, default="ROUND_HALF_UP", help="rounding mode")
    args = p.parse_args()

    print(f"building inputs n={args.n} ...", flush=True)
    a_ints, b_ints, a_sc, b_sc = build_inputs(args.n)

    print("running Cython batch_quantize ...", flush=True)
    cy_out, cy_time = run_cython(a_ints, b_ints, a_sc, b_sc, args.target_scale, args.rounding)
    print(f"cython elapsed {cy_time:.3f}s", flush=True)

    print("running Python quantize_int loop ...", flush=True)
    py_out, py_time = run_python(a_ints, b_ints, a_sc, b_sc, args.target_scale, args.rounding)
    print(f"python elapsed {py_time:.3f}s", flush=True)

    same = True
    if len(cy_out) != len(py_out):
        same = False
    else:
        for i, (x, y) in enumerate(zip(cy_out, py_out)):
            if int(x) != int(y):
                print(f"mismatch at idx {i}: cy={int(x)} py={int(y)}")
                same = False
                break

    print()
    print("results:")
    print(f"  items: {len(cy_out)}")
    print(f"  cython time : {cy_time:.6f}s")
    print(f"  python time : {py_time:.6f}s")
    if py_time > 0:
        print(f"  speedup (python/cython) : {py_time / cy_time:.3f}x")
    else:
        print("  speedup: n/a")
    print(f"  outputs match: {same}")

if __name__ == "__main__":
    main()

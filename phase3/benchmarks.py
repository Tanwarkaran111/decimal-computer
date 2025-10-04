"""
phase3/benchmarks.py

Systematic benchmarking harness for all GEMM implementations discovered
in phase2.* modules.

Outputs:
 - RAW_CSV: raw per-trial timings
 - AGG_CSV: aggregated mean+stdev per (algo,n,digits)
"""
from __future__ import annotations
import argparse
import csv
import importlib
import os
import random
import statistics
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# -----------------------
# Configuration defaults
# -----------------------
DEFAULT_SIZES = [2, 4, 8, 16, 32, 64, 128]
DEFAULT_DIGITS = [1, 2, 4]
DEFAULT_ALGOS = ["auto", "schoolbook", "schoolbook_blocked", "karatsuba", "strassen", "decimal_naive"]

RAW_CSV = "phase3_benchmarks_raw.csv"
AGG_CSV = "phase3_benchmarks_aggregated.csv"
RNG_SEED = 12345

# -----------------------
# Warmup/log helpers
# -----------------------
WARMUP_LOG = Path("phase3_warmup_tracebacks.log")


def _safe_warmup_test(func: Callable) -> bool:
    """Run a few small padded test cases, log full traceback if fails."""
    def _pad_to(n: int) -> List[List[int]]:
        return [[1] * n for _ in range(n)]

    tests = [(_pad_to(2), _pad_to(2)), (_pad_to(4), _pad_to(4))]

    for A, B in tests:
        try:
            try:
                func(A, B)
            except TypeError:
                # try common cutoff arguments as keywords
                tried = False
                for kw in ("base_cutoff", "cutoff", "base"):
                    try:
                        func(A, B, **{kw: 16})
                        tried = True
                        break
                    except TypeError:
                        continue
                if tried:
                    continue
                # try positional third arg as last resort
                func(A, B, 16)
        except Exception:
            with open(WARMUP_LOG, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Warmup fail for {getattr(func, '__name__', repr(func))}\n")
                f.write("Exception traceback:\n")
                traceback.print_exc(file=f)
                f.write("\n")
            return False
    return True


# -----------------------
# Discovery helpers
# -----------------------
def _find_algo_functions() -> Dict[str, Callable]:
    """
    Robust discovery of algorithm callables in phase2.* modules.

    Preference order tries to use public wrappers if available.
    """
    mapping: Dict[str, Callable] = {}
    modules_to_try = [
        "phase2.block_strassen",
        "phase2.parallel_karatsuba",
        "phase2.decimal_gemm",
        "phase2",  # fallback
        "phase6.auto_multiply",      # Phase 6 hybrid wrapper
        "phase4.fft_multiply",       # Phase 4 FFT implementation
        "phase5.ntt_multiply",       # Phase 5 NTT implementation
    ]

    # desired canonical mapping: name -> list of candidates in preference order
    name_map = {
        "strassen": ["block_strassen", "strassen", "_strassen_recursive", "block_strassen_gemm"],
        "karatsuba": ["karatsuba_gemm", "parallel_multiply", "karatsuba", "_karatsuba"],
        "schoolbook": ["_multiply_block_rows", "schoolbook", "schoolbook_gemm", "_schoolbook"],
        "schoolbook_blocked": ["_schoolbook_blocked", "schoolbook_blocked", "_blocked_schoolbook"],
        "decimal_naive": ["decimal_gemm_naive", "decimal_naive", "decimal_gemm"],
        # Phase 6 hybrid auto-multiplier (try these names in phase6.auto_multiply module)
        "auto": ["auto_multiply", "multiply_auto", "auto"],
        # Phase 4 FFT & Phase 5 NTT canonical names
        "fft": ["multiply_ints", "fft_multiply", "multiply_fft", "fft"],
        "ntt": ["multiply_ints_ntt", "ntt_multiply", "multiply_ntt", "ntt"],
    }

    for modname in modules_to_try:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        for want_name, candidates in name_map.items():
            if want_name in mapping:
                # we already resolved this name
                continue
            for cand in candidates:
                fn = getattr(mod, cand, None)
                if callable(fn):
                    mapping[want_name] = fn
                    break

    # Ensure fallback schoolbook exists (try parallel_karatsuba module)
    if "schoolbook" not in mapping:
        try:
            mod = importlib.import_module("phase2.parallel_karatsuba")
            fallback = getattr(mod, "_multiply_block_rows", None)
            if callable(fallback):
                mapping["schoolbook"] = fallback
        except Exception:
            pass

    # ----------------------------
    # If we discovered a scalar multiplier (auto/fft/ntt),
    # adapt it to a GEMM-style interface fn(A, B) -> C so the harness can call it.
    # This adapter is placed here (inside this function) so it can access `mapping`.
    # ----------------------------
    def _make_scalar_gemm(scalar_fn: Callable) -> Callable:
        """Return a tiny GEMM wrapper that uses scalar_fn(a,b,**kwargs) for scalar multiply.

        Optimization: for small integers (fast path) use Python's native a*b.
        If user sets algo-kw 'force_expensive=True', always call scalar_fn.
        """
        # threshold: if both operands are < THRESH in absolute value, use Python multiply
        THRESH = 1 << 30  # ~1e9, adjust if you like

        def gemm(A: List[List[int]], B: List[List[int]], **kwargs):
            if not hasattr(A, "__len__") and not hasattr(B, "__len__"): return A * B
            
            # Optional one-time debug print via scalar_fn(debug=True)
            if kwargs.get("debug", False):
                try:
                    _ = scalar_fn(2, 3, debug=True)
                except Exception:
                    pass

            force_expensive = bool(kwargs.get("force_expensive", False))

            n = len(A)
            p = len(A[0]) if n > 0 else 0
            m = len(B[0]) if p > 0 else 0
            C = [[0 for _ in range(m)] for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    s = 0
                    for k in range(p):
                        a = A[i][k]
                        b = B[k][j]
                        # fast-path for small ints unless forced to use expensive routine
                        if (not force_expensive) and (abs(a) < THRESH) and (abs(b) < THRESH):
                            prod = a * b
                        else:
                            # fall back to the provided scalar function (FFT/NTT/etc.)
                            prod = scalar_fn(a, b, **kwargs)
                        s = s + prod
                    C[i][j] = s
            return C

        return gemm


    for key in ("auto", "fft", "ntt"):
        if key in mapping:
            fn = mapping[key]
            # lightweight check: is this a scalar multiplier? try calling with small ints
            try:
                _ = fn(2, 3)
                is_scalar = True
            except Exception:
                is_scalar = False

            if is_scalar:
                mapping[key] = _make_scalar_gemm(fn)

    return mapping


# -----------------------
# CLI helper: parse algo kwargs
# -----------------------
def parse_algo_kw(s: str) -> dict:
    """
    Parse algo-kw string like: "exact=True,prefer=ntt,base=1000"
    Returns dict with bools/ints/strings parsed.
    """
    out = {}
    if not s:
        return out
    for token in s.split(","):
        if not token.strip():
            continue
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            # try int
            try:
                out[k] = int(v)
            except Exception:
                out[k] = v
    return out

# -----------------------
# Utility: matrix generation
# -----------------------
def gen_matrix(n: int, m: int, digits: int, rng: random.Random) -> List[List[int]]:
    """Generate deterministic random integer matrix with given digits width."""
    maxv = 10 ** digits - 1
    return [[rng.randint(0, maxv) for _ in range(m)] for _ in range(n)]


# -----------------------
# Main runner
# -----------------------
def run_benchmarks(
    sizes: List[int],
    digits_list: List[int],
    algos: List[str],
    trials: int,
    out_raw: str = RAW_CSV,
    out_agg: str = AGG_CSV,
    algo_kwargs: dict = None,
) -> None:
    rng = random.Random(RNG_SEED)
    if algo_kwargs is None:
        algo_kwargs = {}

    funcs = _find_algo_functions()
    resolved_algos: List[Tuple[str, Callable]] = []
    for a in algos:
        if a in funcs:
            resolved_algos.append((a, funcs[a]))
        else:
            print(f"[warning] algorithm '{a}' not found; skipping.")

    if not resolved_algos:
        raise RuntimeError("No algorithms available to benchmark.")

    # Prepare CSV
    with open(out_raw, "w", newline="") as f_raw:
        writer = csv.writer(f_raw)
        writer.writerow(["algo", "size_n", "size_m", "size_p", "digits", "trial", "elapsed_seconds"])
        all_records: List[Tuple[str, int, int, int, int, float]] = []

        for size in sizes:
            n = size
            p = size
            m = size
            for digits in digits_list:
                for trial in range(1, trials + 1):
                    A = gen_matrix(n, p, digits, rng)
                    B = gen_matrix(p, m, digits, rng)

                    for name, func in resolved_algos:
                        # robust warmup test (unchanged)
                        if not _safe_warmup_test(func):
                            print(f"[warning] {name} raised on warmup; skipping this algo for this config.")
                            continue

                        # timed run (single trial measurement)
                        t0 = time.perf_counter()
                        try:
                            # forward algo_kwargs into algorithm call (GEMM wrappers should accept **kwargs)
                            C = func(A, B, **algo_kwargs)
                        except TypeError:
                            # Some older wrappers may not accept kwargs; fallback to simple call
                            C = func(A, B)
                        t1 = time.perf_counter()
                        elapsed = t1 - t0

                        writer.writerow([name, n, m, p, digits, trial, f"{elapsed:.6f}"])
                        all_records.append((name, n, m, p, digits, elapsed))
                        # small flush to keep file safe
                        f_raw.flush()

    # Aggregate (unchanged)
    agg: Dict[Tuple[str, int, int], List[float]] = {}
    for name, n, m, p, digits, elapsed in all_records:
        key = (name, n, digits)
        agg.setdefault(key, []).append(elapsed)

    with open(out_agg, "w", newline="") as fagg:
        w = csv.writer(fagg)
        w.writerow(["algo", "n", "digits", "trials", "mean_seconds", "stdev_seconds"])
        for (name, n, digits), times in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            mean_s = statistics.mean(times)
            stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
            w.writerow([name, n, digits, len(times), f"{mean_s:.6f}", f"{stdev_s:.6f}"])

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run phase3 GEMM benchmarks")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--digits", type=int, nargs="+", default=DEFAULT_DIGITS)
    p.add_argument("--algos", type=str, nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out-raw", type=str, default=RAW_CSV)
    p.add_argument("--out-agg", type=str, default=AGG_CSV)
    p.add_argument("--algo-kw", type=str, default="", help="forwarded key=value pairs to algorithm wrappers, e.g. exact=True,prefer=ntt")
    return p.parse_args()



def main():
    args = parse_args()
    print(f"Running benchmarks with sizes={args.sizes}, digits={args.digits}, algos={args.algos}, trials={args.trials}")
    run_benchmarks(args.sizes, args.digits, args.algos, args.trials, args.out_raw, args.out_agg)
    print(f"Done. Raw CSV: .\\{RAW_CSV} | Aggregated CSV: .\\{AGG_CSV}")


if __name__ == "__main__":
    main()

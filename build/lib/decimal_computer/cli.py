# decimal_computer/cli.py
"""Simple CLI for running decimal GEMM from the command line.

Usage (examples):
    python -m decimal_computer.cli --algo auto --example small --show-counters
    python -m decimal_computer.cli --algo karatsuba --cutoff 16 --example demo
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Tuple, Optional

# relative imports inside package (works with `python -m decimal_computer.cli`)
from .decimal_digit_starter import int_to_digits, digits_to_int
from .decimal_gemm import decimal_gemm_naive

# optional algorithm modules (Strassen / adaptive) — import if available
try:
    from .strassen import strassen_gemm  # type: ignore
except Exception:
    strassen_gemm = None  # type: ignore

try:
    from .adaptive import adaptive_gemm  # type: ignore
except Exception:
    adaptive_gemm = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run decimal GEMM on small example matrices.")
    p.add_argument(
        "--algo",
        choices=["auto", "schoolbook", "karatsuba", "strassen", "adaptive"],
        default="auto",
        help="Which multiplication algorithm to use",
    )
    p.add_argument(
        "--cutoff", type=int, default=None, help="Karatsuba cutoff value (if using karatsuba)"
    )
    p.add_argument(
        "--example",
        choices=["small", "demo"],
        default="small",
        help="Which example matrices to multiply",
    )
    p.add_argument(
        "--show-counters",
        action="store_true",
        help="Print operation counters (muls/adds/time) if available",
    )
    return p.parse_args()


def example_matrix(kind: str):
    """Return two example matrices A,B based on `kind`."""
    if kind == "small":
        # tiny matrices for quick smoke runs
        A = [[12, 34], [56, 78]]
        B = [[87, 65], [43, 21]]
        return A, B
    # slightly larger demo
    A = [[123, 456], [789, 101]]
    B = [[11, 22], [33, 44]]
    return A, B


def _unpack_result(result: Any) -> Tuple[Any, int, int, Optional[float]]:
    """
    Normalize different return shapes into (C, muls, adds, elapsed_seconds).

    Accepts:
      - C
      - (C, counters_dict) where counters has keys "muls","adds","time_s"
      - (C, muls, adds)
      - (C, muls, adds, time_s)

    Always returns (C, muls:int, adds:int, elapsed:float|None)
    """
    if result is None:
        return None, 0, 0, None

    # if result is a tuple/list try to detect shape
    if isinstance(result, (tuple, list)):
        L = list(result)
        # (C, counters_dict)
        if len(L) == 2 and isinstance(L[1], dict):
            C = L[0]
            counters: Dict[str, Any] = L[1]
            muls = int(counters.get("muls", 0))
            adds = int(counters.get("adds", 0))
            elapsed = counters.get("time_s", None)
            return C, muls, adds, float(elapsed) if elapsed is not None else None
        # (C, muls, adds) or (C, muls, adds, time)
        if 2 < len(L) <= 4:
            C = L[0]
            muls = int(L[1]) if len(L) > 1 else 0
            adds = int(L[2]) if len(L) > 2 else 0
            elapsed = float(L[3]) if len(L) > 3 else None
            return C, muls, adds, elapsed
        # (C,) or single element container
        if len(L) == 1:
            return L[0], 0, 0, None

    # anything else: treat as plain C
    return result, 0, 0, None


def run_gemm_dispatch(
    A: Any,
    B: Any,
    algo: str,
    cutoff: Optional[int] = None,
    show_counters: bool = False,
):
    """Dispatch to the requested gemm implementation, returning normalized outputs."""
    # 1) handle special algorithms that are implemented separately
    if algo == "strassen":
        if strassen_gemm is None:
            raise RuntimeError("Strassen implementation not found (strassen.py missing).")
        # strassen_gemm may have different signature; try to call with counters first
        try:
            res = strassen_gemm(A, B, return_counters=True, cutoff=cutoff)
        except TypeError:
            res = strassen_gemm(A, B, cutoff=cutoff)
        return _unpack_result(res)

    if algo == "adaptive":
        if adaptive_gemm is None:
            raise RuntimeError("Adaptive implementation not found (adaptive.py missing).")
        try:
            res = adaptive_gemm(A, B, return_counters=True, cutoff=cutoff)
        except TypeError:
            res = adaptive_gemm(A, B, cutoff=cutoff)
        return _unpack_result(res)

    # default: use decimal_gemm_naive which supports auto/schoolbook/karatsuba
    try:
        # prefer asking for counters in a single call
        res = decimal_gemm_naive(
            A, B, reset_counters_before=True, return_counters=True, mul_algo=algo, cutoff=cutoff
        )
        return _unpack_result(res)
    except TypeError:
        # older signature: call without return_counters then read counters via global counters if available
        start = time.perf_counter()
        maybe = decimal_gemm_naive(A, B, reset_counters_before=True, mul_algo=algo, cutoff=cutoff)
        end = time.perf_counter()
        # try to unpack what we got
        C, muls, adds, _ = _unpack_result(maybe)
        # if muls/adds are zero, we didn't get counters; leave time only
        elapsed = end - start
        return C, muls, adds, elapsed


def main():
    args = parse_args()
    A, B = example_matrix(args.example)

    algo = args.algo
    cutoff = args.cutoff

    print(f"[decimal_gemm] user-selected algo='{algo}' cutoff={cutoff}")

    try:
        C, muls, adds, elapsed = run_gemm_dispatch(A, B, algo=algo, cutoff=cutoff, show_counters=args.show_counters)
    except Exception as e:
        print("ERROR running GEMM:", e)
        raise

    # print the result and counters uniformly
    print(f"[{algo.upper()}] Result: {C}")
    if args.show_counters:
        print(f"digit muls: {muls} digit adds: {adds}")
        if elapsed is not None:
            print(f"Time (s): {elapsed:.6f}")


if __name__ == "__main__":
    main()

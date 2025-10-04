# decimal_computer/adaptive.py
from pathlib import Path
import time
from typing import Tuple, Optional

from .decimal_digit_starter import int_to_digits, digits_to_int  # existing helpers
from .decimal_gemm import decimal_gemm_naive  # existing gemm impl

CUTTABLE_PATH = Path("cutoff_table.json")


def _read_cutoff_table() -> Optional[dict]:
    """Return cutoff table dict or None if not present."""
    if not CUTTABLE_PATH.exists():
        return None
    try:
        import json
        data = json.loads(CUTTABLE_PATH.read_text())
        # convert keys to int
        return {int(k): v for k, v in data.items()}
    except Exception:
        return None


def adaptive_gemm(A: list, B: list, *,
                 max_digit_len: int = 32,
                 default_cutoff: Optional[int] = 16,
                 mul_algo: str = "auto") -> Tuple[list, int, int, float]:
    """
    Run GEMM but pick algorithm adaptively.

    Returns (C, mul_count, add_count, elapsed_seconds)
    - C: result matrix
    - mul_count, add_count: digit operation counters (if gemm returns counters)
    - elapsed_seconds: elapsed wall time
    """
    # choose cutoff per-digit from table if available
    cutoff_table = _read_cutoff_table()
    chosen_cutoff = default_cutoff
    if cutoff_table:
        # determine number of digits per element (rough heuristic)
        # we use first element's digit length as heuristic
        try:
            sample = A[0][0]
            digits = len(int_to_digits(abs(int(sample))))
            # if digits key available, use its cutoff
            entry = cutoff_table.get(digits)
            if entry and "cutoff" in entry:
                chosen_cutoff = int(entry["cutoff"])
        except Exception:
            pass

    # Decide algorithm
    algo = mul_algo
    if mul_algo == "auto":
        # simple rule: use karatsuba if digit size large enough
        if max_digit_len is not None and max_digit_len >= (chosen_cutoff or 16):
            algo = "karatsuba"
        else:
            algo = "schoolbook"

    # Run the gemm implementation (decimal_gemm_naive supports mul_algo, cutoff)
    t0 = time.perf_counter()
    # decimal_gemm_naive returns either C or (C, muls, adds) depending on flags.
    # We call with return_counters True if available
    try:
        C, muls, adds = decimal_gemm_naive(A, B, return_counters=True,
                                          reset_counters_before=True,
                                          mul_algo=algo, karatsuba_cutoff=chosen_cutoff)
    except TypeError:
        # fallback if signature differs: call and do not expect counters
        C = decimal_gemm_naive(A, B, mul_algo=algo, karatsuba_cutoff=chosen_cutoff)
        muls = adds = -1
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return C, muls, adds, elapsed
if __name__ == "__main__":
    from decimal_computer import adaptive_gemm
    A = [[12, 3], [4, 5]]
    B = [[2, 1], [10, 2]]
    print(adaptive_gemm(A, B))

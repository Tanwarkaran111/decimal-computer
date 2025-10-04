from decimal_computer import decimal_gemm_naive
# demo.py  — add these imports (or adapt if already present)
from pathlib import Path
import json
import logging
import time

# import the library function you already used
from decimal_computer.decimal_gemm import decimal_gemm_naive

logging.basicConfig(level=logging.INFO, format="%(message)s")

def _read_cutoff_table() -> dict | None:
    """Read cutoff table created by compute_cutoff_per_digit.py (cutoff_table.json)."""
    p = Path("cutoff_table.json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        # ensure keys are ints (digit lengths) — callers expect numeric keys
        return {int(k): v for k, v in data.items()}
    except Exception:
        return None

def _read_single_cutoff(default: int = 16) -> int:
    """Read single recommended cutoff fallback file."""
    p = Path("recommended_karatsuba_cutoff.txt")
    if p.exists():
        try:
            return int(p.read_text().strip())
        except Exception:
            return default
    return default

def adaptive_gemm(A, B, *, log=True, return_counters=True, **kwargs):
    """
    Run GEMM adaptively:
      - If cutoff_table.json exists use per-digit rules.
      - Else use recommended_karatsuba_cutoff.txt (single integer).
      - Else let decimal_gemm_naive choose heuristically by passing mul_algo="auto".
    Additional kwargs forwarded to decimal_gemm_naive (e.g. mul_algo override).
    Returns: (C, muls, adds, elapsed_s) if return_counters True else C
    """
    # user override: if caller explicitly sets mul_algo then obey
    if "mul_algo" in kwargs and kwargs["mul_algo"] is not None:
        chosen_algo = kwargs["mul_algo"]
        # call directly (we still pass cutoff if provided)
        t0 = time.perf_counter()
        c, muls_adds = decimal_gemm_naive(A, B, return_counters=True, **kwargs)
        t1 = time.perf_counter()
        muls, adds = muls_adds
        if log:
            logging.info(f"[decimal_gemm] user-selected algo={chosen_algo!r} cutoff={kwargs.get('cutoff')}")
        return (c, muls, adds, t1 - t0) if return_counters else c

    # try per-digit table
    table = _read_cutoff_table()
    if table:
        # choose cutoff based on digit-length of input numbers (best-effort)
        # we need max digit length of any entry in A or B: assume matrix elements are int
        def max_digit_len_matrix(M):
            mx = 0
            for row in M:
                for v in row:
                    s = len(str(abs(int(v))))
                    if s > mx:
                        mx = s
            return mx
        max_d = max(max_digit_len_matrix(A), max_digit_len_matrix(B))
        # find closest key in table (if exact key missing)
        if max_d in table:
            best = table[max_d]
        else:
            # choose nearest smaller-or-equal key; otherwise min key
            keys = sorted(table.keys())
            chosen_key = next((k for k in reversed(keys) if k <= max_d), keys[0])
            best = table[chosen_key]

        chosen_algo = best.get("algo", "auto")
        cutoff = best.get("cutoff", None)
        if log:
            logging.info(f"[decimal_gemm] auto-selected algo={chosen_algo!r} max_digit_len={max_d} cutoff={cutoff}")
        t0 = time.perf_counter()
        c, (muls, adds) = decimal_gemm_naive(A, B, mul_algo=chosen_algo, cutoff=cutoff, return_counters=True)
        t1 = time.perf_counter()
        if log:
            logging.info(f"Result: {c}\nTime (s): {t1 - t0:.6f}\ndigit muls: {muls} digit adds: {adds}")
        return (c, muls, adds, t1 - t0) if return_counters else c

    # else fallback to single cutoff file
    cutoff_single = _read_single_cutoff()
    if cutoff_single is not None:
        if log:
            logging.info(f"[decimal_gemm] fallback single cutoff={cutoff_single}, using mul_algo='auto'")
        t0 = time.perf_counter()
        c, (muls, adds) = decimal_gemm_naive(A, B, mul_algo="auto", cutoff=cutoff_single, return_counters=True)
        t1 = time.perf_counter()
        if log:
            logging.info(f"Result: {c}\nTime (s): {t1 - t0:.6f}\ndigit muls: {muls} digit adds: {adds}")
        return (c, muls, adds, t1 - t0) if return_counters else c

    # last fallback: let decimal_gemm_naive decide (auto)
    if log:
        logging.info("[decimal_gemm] no cutoff file found; using mul_algo='auto' default behaviour")
    t0 = time.perf_counter()
    c, (muls, adds) = decimal_gemm_naive(A, B, mul_algo="auto", return_counters=True)
    t1 = time.perf_counter()
    if log:
        logging.info(f"Result: {c}\nTime (s): {t1 - t0:.6f}\ndigit muls: {muls} digit adds: {adds}")
    return (c, muls, adds, t1 - t0) if return_counters else c

# Example matrices
A = [[12, 3], [4, 5]]
B = [[2, 1], [10, 2]]

# Auto-selection
C_auto, muls, adds = decimal_gemm_naive(A, B, return_counters=True, mul_algo="auto")
print("[AUTO] Result:", C_auto, "muls:", muls, "adds:", adds)

# Force schoolbook
C_school, muls_s, adds_s = decimal_gemm_naive(A, B, return_counters=True, mul_algo="schoolbook")
print("[SCHOOLBOOK] Result:", C_school, "muls:", muls_s, "adds:", adds_s)

# Force karatsuba (cutoff = 16)
C_kar, muls_k, adds_k = decimal_gemm_naive(A, B, return_counters=True, mul_algo="karatsuba", karatsuba_cutoff=16)
print("[KARATSUBA] Result:", C_kar, "muls:", muls_k, "adds:", adds_k)

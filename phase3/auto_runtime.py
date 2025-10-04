"""
Auto runtime / selector bridge for GEMM backends.

Behavior:
- Try to pick algorithm from phase3_results/phase3_selector_table.csv (exact match on n+digits,
  else fallback to nearest smaller n with same digits).
- If selector CSV missing, use a simple heuristic: n<32 -> karatsuba, n<128 -> strassen,
  else -> schoolbook_blocked.
- Use tuned cutoffs from phase3_cutoff_table.csv if present (digits -> cutoff mapping).
- When calling schoolbook_blocked, consult phase3_blocksize_table.json for best block size.
- Try various common kw/positional signatures for functions (cutoff, base_cutoff, block size).
"""
import time
import csv
import json
import traceback
from pathlib import Path
from typing import Optional

# --- discovery of implementations (best-effort imports) ---
try:
    from phase2.parallel_karatsuba import karatsuba_gemm as _karatsuba_gemm
except Exception:
    try:
        from phase2.parallel_karatsuba import parallel_multiply as _karatsuba_gemm
    except Exception:
        _karatsuba_gemm = None

try:
    from phase2.block_strassen import block_strassen as _block_strassen
except Exception:
    _block_strassen = None

try:
    # blocked schoolbook implementation
    from phase2.parallel_karatsuba import _multiply_block_rows as _schoolbook_blocked
except Exception:
    _schoolbook_blocked = None

try:
    from phase2.decimal_gemm import decimal_gemm_naive as _decimal_gemm_naive
except Exception:
    _decimal_gemm_naive = None

from phase3.impl_wrappers import WRAPPERS as _WRAPPERS
_ALGOS = {
    "karatsuba": _WRAPPERS["karatsuba"],
    "strassen": _WRAPPERS["strassen"],
    "schoolbook_blocked": _WRAPPERS["schoolbook_blocked"],
    "decimal_naive": _WRAPPERS["decimal_naive"],
}


# --- tuned cutoff table ---
TUNED_CUTOFFS = {}
try:
    cutoff_csv = Path("phase3_cutoff_table.csv")
    if cutoff_csv.exists():
        with cutoff_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    digits = int(row.get("digits") or row.get("d"))
                    cutoff = int(row.get("cutoff") or row.get("best_cutoff") or 64)
                    TUNED_CUTOFFS[(digits,)] = cutoff
                except Exception:
                    continue
except Exception:
    TUNED_CUTOFFS = {}

# --- tuned blocksize table ---
_BLOCKSIZE_TABLE = {}
_blocksize_json = Path("phase3_blocksize_table.json")
try:
    if _blocksize_json.exists():
        j = json.loads(_blocksize_json.read_text())
        tbl = j.get("best_block_for_n", j)
        _BLOCKSIZE_TABLE = {int(k): int(v) for k, v in tbl.items()}
except Exception:
    _BLOCKSIZE_TABLE = {}

# --- selector CSV (auto_selector output) ---
_SELECTOR_CSV = Path("phase3_results/phase3_selector_table.csv")


def _choose_from_selector(n: int, digits: int) -> Optional[str]:
    """Pick algorithm from selector CSV (exact match, else nearest smaller/equal n, else None)."""
    if not _SELECTOR_CSV.exists():
        return None
    rows = []
    try:
        with _SELECTOR_CSV.open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    algo = row["algo"]
                    rn = int(row["n"])
                    rd = int(row["digits"])
                    rows.append((rn, rd, algo))
                except Exception:
                    continue
    except Exception:
        return None

    for rn, rd, algo in rows:
        if rn == n and rd == digits:
            return algo

    candidates = [(rn, algo) for rn, rd, algo in rows if rd == digits and rn <= n]
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    candidates = [(abs(rn - n), rn, algo) for rn, rd, algo in rows if rd == digits]
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    return None


def auto_gemm(A, B, digits: int = 1, debug: bool = True):
    """Auto-choose and run the best GEMM backend."""
    n = len(A)

    # Step 1: try selector table
    name = _choose_from_selector(n, int(digits))

    # Step 2: heuristic fallback
    if name is None:
        if n < 32:
            name = "karatsuba"
        elif n < 128:
            name = "strassen"
        else:
            name = "schoolbook_blocked"

    func = _ALGOS.get(name)
    if debug:
        print(f"[auto_gemm] n={n} digits={digits} -> chosen='{name}' (callable={getattr(func,'__name__',func)})")
    if func is None:
        raise RuntimeError(f"No implementation found for algorithm '{name}'")

    cutoff = TUNED_CUTOFFS.get((int(digits),), 64)

    # Run and time
    t0 = time.perf_counter()
    try:
        if name == "schoolbook_blocked":
            # --- pick best block size ---
            best_b = _BLOCKSIZE_TABLE.get(n)
            if best_b is None:
                smaller = sorted(k for k in _BLOCKSIZE_TABLE if k <= n)
                if smaller:
                    best_b = _BLOCKSIZE_TABLE[smaller[-1]]
                elif _BLOCKSIZE_TABLE:
                    nearest = min(_BLOCKSIZE_TABLE.keys(), key=lambda k: abs(k - n))
                    best_b = _BLOCKSIZE_TABLE[nearest]
                else:
                    best_b = 16

            # try both positional + kw signatures
            try:
                result = func(A, B, best_b)
            except TypeError:
                try:
                    result = func(A, B, block_size=best_b)
                except TypeError:
                    result = func(A, B)

        else:
            try:
                result = func(A, B)
            except TypeError:
                for kw in ("base_cutoff", "cutoff", "base", "base_cutoff_n"):
                    try:
                        result = func(A, B, **{kw: cutoff})
                        break
                    except TypeError:
                        continue
                else:
                    result = func(A, B, cutoff)

    except Exception:
        traceback.print_exc()
        raise
    finally:
        elapsed = time.perf_counter() - t0
        if debug:
            print(f"[auto_gemm] runtime: {elapsed:.6f} seconds")

    return result

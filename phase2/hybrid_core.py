# phase2/hybrid_core.py
"""
Hybrid GEMM Core
----------------
Chooses between Schoolbook, Karatsuba, and Strassen dynamically
based on cutoffs tuned in tuner.py.
"""

import json
import os
import time
from pathlib import Path

# ---------- Safe Import Handling ----------
try:
    from decimal_computer.decimal_gemm import multiply_matrices as schoolbook_gemm
except ImportError:
    try:
        from decimal_computer.decimal_gemm import gemm as schoolbook_gemm
    except ImportError:
        raise ImportError(
            "❌ Could not find a schoolbook multiplication function in decimal_gemm.py.\n"
            "Expected one of: `multiply_matrices` or `gemm`.\n"
            "Please check decimal_gemm.py for the correct function name."
        )

from phase2.parallel_karatsuba import parallel_multiply as karatsuba_gemm
from phase2.block_strassen import block_strassen as strassen_gemm
from phase2.utils import random_matrix

# ---------- Cutoff Handling ----------
CUTOFF_FILE = Path(__file__).parent / "cutoffs.json"

import csv

CUTOFF_FILE = Path(__file__).parent / "cutoffs.json"
CSV_FILE = Path(__file__).parent.parent / "phase3_cutoff_table.csv"

def load_cutoffs():
    # Prefer CSV from Phase 3 tuning
    if CSV_FILE.exists():
        try:
            cutoffs = {}
            with open(CSV_FILE, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    digits = int(row["digits"])
                    cutoff = int(row["suggested_cutoff"])
                    cutoffs[f"schoolbook_vs_strassen_d{digits}"] = cutoff
                if not getattr(load_cutoffs, "_printed", False):
                    print(f"[hybrid_core] Loaded CSV cutoffs: {cutoffs}")
                    load_cutoffs._printed = True
                return cutoffs
        except Exception as e:
            print(f"[hybrid_core] Failed to load CSV cutoffs: {e}")

    # Fall back to JSON if no CSV
    if CUTOFF_FILE.exists():
        try:
            with open(CUTOFF_FILE, "r") as f:
                cutoffs = json.load(f)
                if not getattr(load_cutoffs, "_printed", False):
                    print(f"[hybrid_core] Loaded JSON cutoffs: {cutoffs}")
                    load_cutoffs._printed = True
                return cutoffs
        except Exception as e:
            print(f"[hybrid_core] Failed to load JSON cutoffs: {e}")

    return {}


def save_cutoffs(cutoffs: dict):
    with open(CUTOFF_FILE, "w") as f:
        json.dump(cutoffs, f, indent=2)

cutoff = TUNED_CUTOFFS.get((digits,), DEFAULT_CUTOFFS.get((digits,), 64))


# ---------- Core Hybrid GEMM ----------
def hybrid_gemm(A, B, digits=8, verbose=True):
    """
    Hybrid GEMM:
    - Uses Schoolbook for small n
    - Switches to Karatsuba after cutoff
    - Switches to Strassen for very large n
    """
    n = len(A)
    key_sk = f"schoolbook_vs_karatsuba_d{digits}"
    key_ks = f"karatsuba_vs_strassen_d{digits}"

    cutoff_sk = CUTOFFS.get(key_sk, 32)
    cutoff_ks = CUTOFFS.get(key_ks, 128)

    if n < cutoff_sk:
        algo = "schoolbook"
        fn = schoolbook_gemm
    elif n < cutoff_ks:
        algo = "karatsuba"
        fn = karatsuba_gemm
    else:
        algo = "strassen"
        fn = strassen_gemm

    if verbose:
        print(f"[hybrid_core] hybrid_gemm: using {algo} (n={n}, digits={digits})")

    t0 = time.perf_counter()
    result = fn(A, B)
    elapsed = time.perf_counter() - t0

    return result, {"algo": algo, "elapsed": elapsed, "n": n, "digits": digits}

# ---------- Standalone Test ----------
if __name__ == "__main__":
    A = random_matrix(32, 8)
    B = random_matrix(32, 8)
    C, meta = hybrid_gemm(A, B, digits=8)
    print("meta:", meta)
    print("top-left:", [row[:4] for row in C[:4]])

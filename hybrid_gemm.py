import json
from pathlib import Path

from decimal_computer.decimal_gemm import schoolbook_gemm
from decimal_computer.karatsuba import multiply_matrices as karatsuba_gemm
from decimal_computer.strassen import strassen_gemm

# Load cutoffs from JSON if available
CUTOFFS_PATH = Path("D:/Invented_Library/recommended_cutoffs.json")
if CUTOFFS_PATH.exists():
    with open(CUTOFFS_PATH) as f:
        RECOMMENDED_CUTOFFS = json.load(f)
else:
    RECOMMENDED_CUTOFFS = {}

def hybrid_gemm(A, B, digits: int, size: int):
    """
    Adaptive GEMM selector:
    Uses benchmark-derived cutoffs from JSON if available,
    otherwise falls back to default heuristic.
    """

    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(f"Dimension mismatch: A is {rows_A}x{cols_A}, B is {rows_B}x{cols_B}")

    # --- Use benchmark-driven cutoffs if available ---
    if str(digits) in RECOMMENDED_CUTOFFS:
        rules = RECOMMENDED_CUTOFFS[str(digits)]
        k_cut = rules.get("karatsuba")
        s_cut = rules.get("strassen")

        if k_cut and size >= k_cut and (not s_cut or size < s_cut):
            return karatsuba_gemm(A, B)
        elif s_cut and size >= s_cut:
            return strassen_gemm(A, B)
        else:
            return schoolbook_gemm(A, B)

    # --- Fallback heuristic if no JSON ---
    if size < 8 and digits < 8:
        return schoolbook_gemm(A, B)
    elif size < 64:
        return karatsuba_gemm(A, B)
    else:
        return strassen_gemm(A, B)

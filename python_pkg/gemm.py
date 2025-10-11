import os
import ctypes
from pathlib import Path
import numpy as np
# === Safe-load DLL (copy to temp before loading) ===
_lib_path = Path(__file__).resolve().parent.parent / "build" / "gemm_native_packed.dll"

# --- 🔧 Fix: ensure OpenMP runtime (libgomp) is visible to Python ---
ucrt64_bin = Path("D:/ucrt64/bin")
os.add_dll_directory(str(ucrt64_bin))  # Windows 10+ API for DLL search path

# Now load the GEMM DLL
_lib = ctypes.CDLL(str(_lib_path))

print(f"✅ Loaded GEMM DLL: {_lib_path}")
# === Define the C function signature ===
from ctypes import c_int, c_double, POINTER

_lib.gemm_dispatch_packed.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int, c_int, c_int,
    c_int, c_int, c_int
]
_lib.gemm_dispatch_packed.restype = None


# === Python wrapper for packed GEMM ===
def gemm_packed(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication using the packed-B AVX2 micro-kernel.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = np.zeros((M, N), dtype=np.float64)

    _lib.gemm_dispatch_packed(
        A.ctypes.data_as(POINTER(c_double)),
        B.ctypes.data_as(POINTER(c_double)),
        C.ctypes.data_as(POINTER(c_double)),
        c_int(M), c_int(N), c_int(K),
        c_int(lda), c_int(ldb), c_int(ldc)
    )
    return C

# === Optional quick test ===
if __name__ == "__main__":
    A = np.random.randn(128, 128)
    B = np.random.randn(128, 128)
    C = gemm_packed(A, B)
    print("Diff vs NumPy:", np.max(np.abs(C - A.dot(B))))

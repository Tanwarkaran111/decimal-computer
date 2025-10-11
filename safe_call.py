# safe_call.py
import numpy as np, ctypes, os, time, sys, struct
os.add_dll_directory(r"D:\ucrt64\bin")

DLL = r"D:\Invented_library\build\gemm_native_packed.dll"
lib = ctypes.CDLL(DLL)

# Use size_t for dims (safe on x64)
size_t = ctypes.c_size_t
lib.gemm_dispatch_packed.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
lib.gemm_dispatch_packed.restype = None


def run_once(M,N,K):
    A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float64))
    B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float64))
    C = np.ascontiguousarray(np.zeros((M, N), dtype=np.float64))
    # print addresses for debugging
    # print("A,B,C addrs:", A.ctypes.data, B.ctypes.data, C.ctypes.data)
    lib.gemm_dispatch_packed(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        C.ctypes.data_as(ctypes.c_void_p),
        size_t(M), size_t(N), size_t(K),
        size_t(K), size_t(N), size_t(N)
    )

if __name__ == "__main__":
    import random
    M,N,K = 512,512,512
    # repeat many times to try to trigger the intermittent crash
    for i in range(200):
        try:
            start = time.perf_counter()
            run_once(M,N,K)
            elapsed = time.perf_counter() - start
            print(f"iter {i:03d} OK (time {elapsed:.3f}s)")
        except OSError as e:
            print(f"iter {i:03d} OSError (access violation):", e)
            break
        except Exception as e:
            print(f"iter {i:03d} Exception:", type(e), e)
            break

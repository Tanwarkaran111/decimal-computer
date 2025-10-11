# thread_sweep.py
import ctypes, numpy as np, os, time, struct, sys, traceback

DLL = r"D:\Invented_library\build\gemm_native_packed.dll"
os.add_dll_directory(r"D:\ucrt64\bin")

print("Python bits:", struct.calcsize("P")*8)
lib = ctypes.CDLL(DLL)

lib.gemm_dispatch_packed.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
lib.gemm_dispatch_packed.restype = None

M,N,K = 512,512,512  # use a size that previously crashed
for threads in (1,2,4,8):
    print("\n--- threads =", threads, "---")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float64))
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float64))
        C = np.ascontiguousarray(np.zeros((M, N), dtype=np.float64))
        t0 = time.perf_counter()
        lib.gemm_dispatch_packed(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(M), ctypes.c_size_t(N), ctypes.c_size_t(K),
            ctypes.c_size_t(K), ctypes.c_size_t(N), ctypes.c_size_t(N)
        )
        t1 = time.perf_counter()
        print("OK (time {:.4f}s)".format(t1-t0))
    except Exception as e:
        print("FAILED with exception:", type(e), e)
        traceback.print_exc()

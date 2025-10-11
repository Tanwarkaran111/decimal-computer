# small_probe.py — tiny, very-informative smoke test
import ctypes, numpy as np, os, time, struct, sys, traceback

DLL = r"D:\Invented_library\build\gemm_native_packed.dll"
os.add_dll_directory(r"D:\ucrt64\bin")

print("Python bits:", struct.calcsize("P")*8)
print("Loading DLL:", DLL)
lib = ctypes.CDLL(DLL)

# binding: use size_t for dims/strides
lib.gemm_dispatch_packed.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
lib.gemm_dispatch_packed.restype = None

def info_arr(name, a):
    print(f"{name}: dtype={a.dtype} shape={a.shape} c_contig={a.flags['C_CONTIGUOUS']} "
          f"data_ptr={a.ctypes.data} first_elems={a.ravel()[:8].tolist()}")

# small sizes
M, N, K = 8, 8, 8
A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float64))
B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float64))
C = np.ascontiguousarray(np.zeros((M, N), dtype=np.float64))

print("Arrays created.")
info_arr("A", A)
info_arr("B", B)
info_arr("C", C)

lda = K   # row-major A: stride is K
ldb = N   # row-major B: stride is N
ldc = N   # C row-major: stride is N

print("Calling with M,N,K, lda,ldb,ldc =", M, N, K, lda, ldb, ldc)
try:
    t0 = time.perf_counter()
    lib.gemm_dispatch_packed(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        C.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(M), ctypes.c_size_t(N), ctypes.c_size_t(K),
        ctypes.c_size_t(lda), ctypes.c_size_t(ldb), ctypes.c_size_t(ldc)
    )
    t1 = time.perf_counter()
    print("Call returned OK (time {:.6f}s)".format(t1 - t0))
    info_arr("C(after)", C)
except Exception as e:
    print("Python caught exception:", type(e), e)
    traceback.print_exc()
    # show last 64 bytes of memory around pointers (best-effort, may segfault if invalid)
    try:
        import ctypes as _ct
        def peek(addr, n=64):
            buf = (ctypes.c_ubyte * n).from_address(addr)
            return bytes(buf[:n])
        print("A ptr peek (first 64 bytes):", peek(A.ctypes.data, 64))
        print("B ptr peek (first 64 bytes):", peek(B.ctypes.data, 64))
        print("C ptr peek (first 64 bytes):", peek(C.ctypes.data, 64))
    except Exception as ex2:
        print("Peek failed:", ex2)
    sys.exit(1)

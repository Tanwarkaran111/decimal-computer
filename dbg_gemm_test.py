# dbg_gemm_test.py
import sys, time, traceback, numpy as np, os
sys.path.insert(0, "src")

def info_arr(name, a):
    print(f"{name}: shape={a.shape} dtype={a.dtype} c_contig={a.flags['C_CONTIGUOUS']} f_contig={a.flags['F_CONTIGUOUS']} strides={a.strides}")

try:
    import myext, myext.helpers as helpers
    _g = getattr(helpers, "_g", None)
    print("myext file:", getattr(myext, "__file__", None))
    print("helpers._g type:", type(_g), getattr(_g, "__file__", None))
    print("has gemm_openmp:", bool(_g and hasattr(_g, "gemm_openmp")))
    print("has gemm_block :", bool(_g and hasattr(_g, "gemm_block")))
    print()

    # small reproducible matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    print("=== small integer test (2x2) ===")
    info_arr("A", A); info_arr("B", B)
    C = myext.gemm(A, B)
    print("C from myext.gemm:", C)
    print("np.dot(A,B):", A.dot(B))
    print("max error:", np.max(np.abs(C - A.dot(B))))
    print()

    # explicit call to native entrypoints (if available)
    if _g is not None:
        print("=== direct native calls (explicit) ===")
        C0 = np.zeros((2,2), dtype=np.float64)
        try:
            if hasattr(_g, "gemm_openmp"):
                print("calling _g.gemm_openmp(A,B,C0,2,2,2)")
                _g.gemm_openmp(A, B, C0, 2, 2, 2)
                print("C0:", C0, "err:", np.max(np.abs(C0 - A.dot(B))))
            if hasattr(_g, "gemm_block"):
                C1 = np.zeros((2,2), dtype=np.float64)
                print("calling _g.gemm_block(A,B,C1,0,0,0,2,2,2)  # block covering full matrix")
                _g.gemm_block(A, B, C1, 0, 0, 0, 2, 2, 2)
                print("C1:", C1, "err:", np.max(np.abs(C1 - A.dot(B))))
        except Exception:
            print("Exception from native direct calls:")
            traceback.print_exc()

    print()
    # test contiguous / transposed / non-contiguous inputs
    print("=== contiguity tests ===")
    for kind, AA in [("C-contig", A.copy()), ("F-contig", np.asfortranarray(A)), ("slice non-contig", (A.copy()[:, ::1])[::1])]:
        for kindB, BB in [("C-contig", B.copy()), ("F-contig", np.asfortranarray(B))]:
            Cx = np.zeros((2,2), dtype=np.float64)
            info = f"A_kind={kind} B_kind={kindB}"
            print(info)
            info_arr("AA", AA); info_arr("BB", BB)
            # call wrapper
            Cr = myext.gemm(AA, BB)
            print("wrapper err:", np.max(np.abs(Cr - AA.dot(BB))))
            # if direct native available, try gemm_block with explicit Cx
            if _g is not None and hasattr(_g, "gemm_block"):
                try:
                    _g.gemm_block(AA, BB, Cx, 0, 0, 0, 2, 2, 2)
                    print("gemm_block err:", np.max(np.abs(Cx - AA.dot(BB))))
                except Exception:
                    print("gemm_block raised:")
                    traceback.print_exc()
            print()

    # timing + larger random test
    N = 128
    A2 = np.random.rand(N,N).astype(np.float64)
    B2 = np.random.rand(N,N).astype(np.float64)
    t0 = time.perf_counter()
    C2 = myext.gemm(A2, B2)
    t1 = time.perf_counter()
    print("large test time:", t1-t0, "shape:", getattr(C2, "shape", None))
    print("large test max error:", np.max(np.abs(C2 - A2.dot(B2))))

except Exception:
    traceback.print_exc()
    sys.exit(1)

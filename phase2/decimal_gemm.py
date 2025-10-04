# phase2/decimal_gemm.py
# Small compatibility shim so other tools can import `decimal_gemm_naive` or `decimal_naive`.
# This calls into your existing schoolbook/parallel multiply if available, otherwise uses
# a local simple fallback.

def _fallback_schoolbook(A, B):
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0]*m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik = A[i][k]
            if aik == 0:
                continue
            rowi = C[i]
            brow = B[k]
            for j in range(m):
                rowi[j] += aik * brow[j]
    return C

# Try to wire to any existing function in phase2 that looks appropriate.
try:
    # if you have a decimal-specific routine, this will import it
    from .decimal_gemm_impl import decimal_gemm_naive as decimal_gemm_naive  # optional user file
except Exception:
    try:
        # fallback: parallel_karatsuba exposes a block multiply we discovered earlier
        from .parallel_karatsuba import karatsuba_gemm as _kar_gemm
        # expose both names expected by the benchmark
        def decimal_gemm_naive(A, B):
            return _kar_gemm(A, B)
    except Exception:
        # last-resort fallback
        def decimal_gemm_naive(A, B):
            return _fallback_schoolbook(A, B)

# Also provide the alternate alias 'decimal_naive' used by the CLI defaults
decimal_naive = decimal_gemm_naive

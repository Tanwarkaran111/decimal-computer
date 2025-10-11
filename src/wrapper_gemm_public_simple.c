// wrapper_gemm_public_simple.c
// Simple public wrapper that forwards calls directly to exported
// gemm_dispatch_packed in the existing DLL. Uses double precision.

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <malloc.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*gemm_dispatch_packed_t)(const double* A, const double* B, double* C,
                                       size_t M, size_t N, size_t K,
                                       size_t lda, size_t ldb, size_t ldc);

static const char *TARGET_DLL = "D:\\Invented_library\\build\\gemm_native_packed_stable320.dll";
static HMODULE h = NULL;
static gemm_dispatch_packed_t gemmPacked = NULL;

static int ensure_loaded(void) {
    if (gemmPacked) return 1;
    if (!h) h = LoadLibraryA(TARGET_DLL);
    if (!h) return 0;
    gemmPacked = (gemm_dispatch_packed_t)GetProcAddress(h, "gemm_dispatch_packed");
    return gemmPacked != NULL;
}

__declspec(dllexport) int gemm_native_public_simple(
    const double *A, const double *B, double *C,
    int M, int N, int K
) {
    if (!ensure_loaded()) return 0;

    size_t lda = (size_t)M;   // Fortran-order column-major A has leading-dim = rows = M
    size_t ldb = (size_t)K;   // B is K x N, leading dim (rows) = K
    size_t ldc = (size_t)M;   // C is M x N, leading dim = M

    // call the internal dispatcher which will do its own packing & blocking
    gemmPacked(A, B, C, (size_t)M, (size_t)N, (size_t)K, lda, ldb, ldc);
    return 1;
}

#ifdef __cplusplus
}
#endif

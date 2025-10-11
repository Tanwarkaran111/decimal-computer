// wrapper_packed_final.c
// Matches exactly the kernel's pack_B_block_padded logic (double precision).

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*gemm_dispatch_packed_t)(
    const void* A, const void* B_packed, void* C,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc
);

static const char *TARGET_DLL = "D:\\Invented_library\\build\\gemm_native_packed_stable320.dll";
static HMODULE cached_h = NULL;
static gemm_dispatch_packed_t cached_fn = NULL;

static int ensure_loaded(void) {
    if (cached_fn) return 1;
    if (!cached_h) {
        cached_h = LoadLibraryA(TARGET_DLL);
        if (!cached_h) return 0;
    }
    cached_fn = (gemm_dispatch_packed_t)GetProcAddress(cached_h, "gemm_dispatch_packed");
    return cached_fn != NULL;
}

// Pad K up to multiple of MIC_K_TILE = 4 (same as in kernel)
static inline int round_up_int(int x, int r) {
    if (r <= 0) return x;
    return ((x + r - 1) / r) * r;
}

__declspec(dllexport) int gemm_native_packed_final(
    const double* A, const double* B, double* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return 0;

    const int MIC_K_TILE = 4;
    int klen_padded = round_up_int(K, MIC_K_TILE);
    size_t total = (size_t)klen_padded * (size_t)N;
    double* Bpacked = (double*) malloc(total * sizeof(double));
    if (!Bpacked) return 0;

    // pack columns of B (Fortran/column-major style)
    for (int j = 0; j < N; ++j) {
        double *dst = Bpacked + (size_t)j * (size_t)klen_padded;
        const double *src_col = B + (size_t)j * (size_t)ldb;
        for (int k = 0; k < K; ++k) {
            dst[k] = src_col[k];
        }
        for (int k = K; k < klen_padded; ++k) {
            dst[k] = 0.0;
        }
    }

    cached_fn(A, Bpacked, C, M, N, K, lda, ldb, ldc);
    free(Bpacked);
    return 1;
}

#ifdef __cplusplus
}
#endif

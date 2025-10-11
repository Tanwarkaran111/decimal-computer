// wrapper_packed_both.c
// Perfectly matches your kernel's internal packing format (MIC_N_TILE=8, MIC_K_TILE=4)
// Packs both A and B, then calls gemm_dispatch_packed() from the main DLL.

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*gemm_dispatch_packed_t)(
    const void* A, const void* B, void* C,
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

static inline int round_up_int(int x, int r) {
    return ((x + r - 1) / r) * r;
}

#define MIC_M_TILE 8
#define MIC_N_TILE 8
#define MIC_K_TILE 4

__declspec(dllexport) int gemm_native_packed_both(
    const double* A, const double* B, double* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return 0;

    // Pad M, N, K to tile sizes
    int M_pad = round_up_int(M, MIC_M_TILE);
    int N_pad = round_up_int(N, MIC_N_TILE);
    int K_pad = round_up_int(K, MIC_K_TILE);

    // Allocate packed buffers
    double* A_packed = (double*) malloc((size_t)M_pad * (size_t)K_pad * sizeof(double));
    double* B_packed = (double*) malloc((size_t)K_pad * (size_t)N_pad * sizeof(double));
    if (!A_packed || !B_packed) {
        free(A_packed);
        free(B_packed);
        return 0;
    }

    // ---- Pack A (row-major style, pad K to MIC_K_TILE) ----
    for (int i = 0; i < M; ++i) {
        double *dst = A_packed + (size_t)i * (size_t)K_pad;
        const double *src = A + (size_t)i * (size_t)lda;
        memcpy(dst, src, (size_t)K * sizeof(double));
        for (int k = K; k < K_pad; ++k) dst[k] = 0.0;
    }
    // pad remaining rows of A
    for (int i = M; i < M_pad; ++i) {
        double *dst = A_packed + (size_t)i * (size_t)K_pad;
        for (int k = 0; k < K_pad; ++k) dst[k] = 0.0;
    }

    // ---- Pack B (column-major style, pad K to MIC_K_TILE) ----
    for (int j = 0; j < N; ++j) {
        double *dst = B_packed + (size_t)j * (size_t)K_pad;
        const double *src_col = B + (size_t)j * (size_t)ldb;
        for (int k = 0; k < K; ++k) dst[k] = src_col[k];
        for (int k = K; k < K_pad; ++k) dst[k] = 0.0;
    }
    // pad extra columns of B
    for (int j = N; j < N_pad; ++j) {
        double *dst = B_packed + (size_t)j * (size_t)K_pad;
        for (int k = 0; k < K_pad; ++k) dst[k] = 0.0;
    }

    // ---- Call kernel ----
    cached_fn(A_packed, B_packed, C, M, N, K, lda, ldb, ldc);

    free(A_packed);
    free(B_packed);
    return 1;
}

#ifdef __cplusplus
}
#endif

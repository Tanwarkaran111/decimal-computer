// wrapper_packed_final2.c
// Final wrapper that packs A and B according to kernel_wrapper_packed.c
// and calls gemm_dispatch_packed. Double precision.

#include <windows.h>
#include <malloc.h>    // for _aligned_malloc/_aligned_free on Windows
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*gemm_dispatch_packed_t)(
    const double* A_packed, const double* B_packed, double* C,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc
);

static const char *TARGET_DLL = "D:\\Invented_library\\build\\gemm_native_packed_stable320.dll";
static HMODULE cached_h = NULL;
static gemm_dispatch_packed_t cached_fn = NULL;

// Keep in sync with kernel_wrapper_packed.c
#ifndef MIC_M_TILE
#define MIC_M_TILE 8
#endif
#ifndef MIC_N_TILE
#define MIC_N_TILE 8
#endif
#ifndef MIC_K_TILE
#define MIC_K_TILE 4
#endif

static inline int round_up_int_local(int x, int r) {
    if (r <= 0) return x;
    return ((x + r - 1) / r) * r;
}

static inline void *aligned_alloc_local(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    return p;
#endif
}
static inline void aligned_free_local(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static int ensure_loaded(void) {
    if (cached_fn) return 1;
    if (!cached_h) {
        // ensure loader can find UCRT/GOMP if needed
        AddDllDirectory(L"D:\\ucrt64\\bin");
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        cached_h = LoadLibraryA(TARGET_DLL);
        if (!cached_h) return 0;
    }
    cached_fn = (gemm_dispatch_packed_t)GetProcAddress(cached_h, "gemm_dispatch_packed");
    return cached_fn != NULL;
}

/*
 Public API:
 - A: M x K (Fortran/column-major expected by caller)
 - B: K x N (Fortran/column-major expected by caller)
 - C: M x N (Fortran/column-major)
 We'll pack A into A_packed as M_pad x K_pad, row-major blocks (per-row contiguous K_pad),
 and B into B_packed as K_pad x N_pad, with each column padded to K_pad.
*/

__declspec(dllexport) int gemm_native_packed_final2(
    const double* A, const double* B, double* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return 0;

    int K_pad = round_up_int_local(K, MIC_K_TILE);
    int M_pad = round_up_int_local(M, MIC_M_TILE);
    int N_pad = round_up_int_local(N, MIC_N_TILE);

    // allocate packed buffers: A_packed size = M_pad * K_pad, B_packed size = K_pad * N_pad
    size_t A_elems = (size_t)M_pad * (size_t)K_pad;
    size_t B_elems = (size_t)K_pad * (size_t)N_pad;

    double *A_packed = (double*) aligned_alloc_local(64, A_elems * sizeof(double));
    double *B_packed = (double*) aligned_alloc_local(64, B_elems * sizeof(double));
    if (!A_packed || !B_packed) {
        if (A_packed) aligned_free_local(A_packed);
        if (B_packed) aligned_free_local(B_packed);
        return 0;
    }

    // Zero buffers first (defensive, matches kernel style)
    memset(A_packed, 0, A_elems * sizeof(double));
    memset(B_packed, 0, B_elems * sizeof(double));

    // Pack A: for each row i (0..M-1), copy K elements from A row (A is column-major so stride lda)
    // We assume caller provides A in column-major layout: A[ row + col*lda ].
    for (int i = 0; i < M; ++i) {
        double *dst = A_packed + (size_t)i * (size_t)K_pad;
        // src row i in column-major: element at (i, k) is A[i + k*lda]
        const double *src = A + (size_t)i;
        for (int k = 0; k < K; ++k) {
            dst[k] = src[(size_t)k * (size_t)lda];
        }
        // remaining k in K_pad already zeroed
    }
    // pad remaining rows (already zeroed)

    // Pack B: for each column j (0..N-1), copy K elements from B column (B column-major stride ldb)
    for (int j = 0; j < N; ++j) {
        double *dst = B_packed + (size_t)j * (size_t)K_pad;
        const double *src_col = B + (size_t)j * (size_t)ldb;
        for (int k = 0; k < K; ++k) {
            dst[k] = src_col[k];
        }
        // remaining padded entries already zeroed
    }
    // pad remaining columns (already zeroed)

    // Call the internal dispatcher:
    // Note: pass lda, ldb, ldc as given (the dispatcher uses those for logging only)
    cached_fn(A_packed, B_packed, C,
              (size_t)M, (size_t)N, (size_t)K,
              (size_t)lda, (size_t)ldb, (size_t)ldc);

    aligned_free_local(A_packed);
    aligned_free_local(B_packed);
    return 1;
}

#ifdef __cplusplus
}
#endif

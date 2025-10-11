// wrapper_packed_final3.c
// Final working wrapper for gemm_dispatch_packed
// A is passed as-is; only B is packed exactly as in kernel_wrapper_packed.c

#include <windows.h>
#include <malloc.h>   // for _aligned_malloc/_aligned_free
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*gemm_dispatch_packed_t)(
    const double* A, const double* B_packed, double* C,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc
);

static const char *TARGET_DLL = "D:\\Invented_library\\build\\gemm_native_packed_stable320.dll";
static HMODULE cached_h = NULL;
static gemm_dispatch_packed_t cached_fn = NULL;

// micro-tile sizes (from kernel_wrapper_packed.c)
#ifndef MIC_N_TILE
#define MIC_N_TILE 8
#endif
#ifndef MIC_K_TILE
#define MIC_K_TILE 4
#endif

static inline int round_up_int_local(int x, int r) {
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
        AddDllDirectory(L"D:\\ucrt64\\bin");
        SetDefaultDllDirectories((DWORD)(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS));
        cached_h = LoadLibraryA(TARGET_DLL);
        if (!cached_h) {
            fprintf(stderr, "Failed to load target DLL\n");
            return 0;
        }
    }
    cached_fn = (gemm_dispatch_packed_t)GetProcAddress(cached_h, "gemm_dispatch_packed");
    if (!cached_fn) {
        fprintf(stderr, "Failed to find gemm_dispatch_packed in target DLL\n");
        return 0;
    }
    return 1;
}

// --- B PACKER --- (copied from kernel_wrapper_packed.c)
static void pack_B_block_padded(const double *B, int ldb,
                                int K_tile, int klen_padded, int N_tile,
                                double *B_packed, size_t per_buf_elems)
{
    for (int j = 0; j < N_tile; ++j) {
        double *dst = B_packed + j * klen_padded;
        const double *src_col = B + j;
        for (int k = 0; k < K_tile; ++k) {
            dst[k] = src_col[k * ldb];
        }
        for (int k = K_tile; k < klen_padded; ++k) {
            dst[k] = 0.0;
        }
    }
}

// --- PUBLIC ENTRY ---
__declspec(dllexport) int gemm_native_packed_final3(
    const double* A, const double* B, double* C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
   
{
    

    if (!ensure_loaded()) return 0;

    int K_pad = round_up_int_local(K, MIC_K_TILE);
    int N_pad = round_up_int_local(N, MIC_N_TILE);

    // Allocate packed B
    size_t B_elems = (size_t)K_pad * (size_t)N_pad;
    double *B_packed = (double*)aligned_alloc_local(64, B_elems * sizeof(double));
    if (!B_packed) return 0;
    memset(B_packed, 0, B_elems * sizeof(double));

    // Pack B in tiles of MIC_N_TILE columns
    for (int j = 0; j < N; j += MIC_N_TILE) {
        int N_tile = (N - j < MIC_N_TILE) ? (N - j) : MIC_N_TILE;
        pack_B_block_padded(B + (size_t)j * (size_t)ldb, ldb,
                            K, K_pad, N_tile,
                            B_packed + (size_t)j * (size_t)K_pad,
                            B_elems);
    }

    // Call kernel
    cached_fn(A, B_packed, C,
              (size_t)M, (size_t)N, (size_t)K,
              (size_t)lda, (size_t)ldb, (size_t)ldc);

    aligned_free_local(B_packed);
    return 1;
}

#ifdef __cplusplus
}
#endif

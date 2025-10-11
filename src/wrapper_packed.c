// wrapper_packed.c
// Packs B into a simple BLOCK-K x BLOCK-N panel-major layout and calls gemm_dispatch_packed
// Note: tweak BLOCK_K / BLOCK_N if you know different values.

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

// Set the path to the existing DLL that contains gemm_dispatch_packed
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

/* Simple pack: pack B by panels of (BLOCK_K x BLOCK_N). We copy B's column-major (Fortran) layout
   into a packed buffer where each panel stores BLOCK_K rows × BLOCK_N cols contiguously,
   scanning panels left-to-right (n-block) then top-to-bottom (k-block).
   This is a common packed-B layout expected by many kernels. */

#define BLOCK_K 128
#define BLOCK_N 256

__declspec(dllexport) void gemm_native_packed_pack_then_run(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return;

    // compute number of panels
    int n_panels = (N + BLOCK_N - 1) / BLOCK_N;
    int k_panels = (K + BLOCK_K - 1) / BLOCK_K;

    // allocate packed buffer: total size = K * N (floats) but we store per-panel contiguous
    size_t total_elems = (size_t)K * (size_t)N;
    float* Bpacked = (float*) malloc(total_elems * sizeof(float));
    if (!Bpacked) return;

    float* out = Bpacked;
    // assume B is column-major (Fortran) with leading dim ldb (rows = K)
    for (int np = 0; np < n_panels; ++np) {
        int n0 = np * BLOCK_N;
        int nb = (N - n0) < BLOCK_N ? (N - n0) : BLOCK_N;
        for (int kp = 0; kp < k_panels; ++kp) {
            int k0 = kp * BLOCK_K;
            int kb = (K - k0) < BLOCK_K ? (K - k0) : BLOCK_K;
            // for this (kp,np) panel, copy kb x nb block from B (col-major)
            for (int col = 0; col < nb; ++col) {
                int col_idx = n0 + col;
                const float* src_col = B + (size_t)col_idx * (size_t)ldb + (size_t)k0;
                // copy kb elements from src_col into out
                memcpy(out, src_col, kb * sizeof(float));
                out += kb;
                // if kb < BLOCK_K, pad the rest of the panel row with zeros so kernel can read full panel
                if (kb < BLOCK_K) {
                    size_t pad = (size_t)(BLOCK_K - kb);
                    memset(out, 0, pad * sizeof(float));
                    out += pad;
                }
            }
            // if nb < BLOCK_N, pad remaining columns
            if (nb < BLOCK_N) {
                int pad_cols = BLOCK_N - nb;
                // each pad column occupies BLOCK_K floats (already padded to BLOCK_K above)
                memset(out, 0, (size_t)pad_cols * (size_t)BLOCK_K * sizeof(float));
                out += (size_t)pad_cols * (size_t)BLOCK_K;
            }
        }
    }

    // call the packed dispatcher
    cached_fn((const void*)A, (const void*)Bpacked, (void*)C,
              (size_t)M, (size_t)N, (size_t)K,
              (size_t)lda, (size_t)ldb, (size_t)ldc);

    free(Bpacked);
}

#ifdef __cplusplus
}
#endif

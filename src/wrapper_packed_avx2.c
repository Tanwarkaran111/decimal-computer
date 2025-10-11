// wrapper_packed_avx2.c
// Pack B into micro-panels of MR=8 (interleaved 8-row micro-panels)
// and call gemm_dispatch_packed from the existing DLL.
//
// Build:
// gcc -shared -O3 -fPIC -I src src/wrapper_packed_avx2.c -o build/gemm_native_packed_avx2.dll -Wl,--export-all-symbols

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

/* Tunable blocking parameters. Keep BLOCK_K/BLOCK_N reasonably large, and MR
   small (micro-panel height) — many AVX2 kernels use MR=8 (8 floats per vector lane). */
#define BLOCK_K 128
#define BLOCK_N 256
#define MR 8

__declspec(dllexport) int gemm_native_packed_pack_then_run_avx2(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return 0;

    int n_panels = (N + BLOCK_N - 1) / BLOCK_N;
    int k_panels = (K + BLOCK_K - 1) / BLOCK_K;

    size_t total_elems = (size_t)K * (size_t)N;
    float* Bpacked = (float*) malloc(total_elems * sizeof(float));
    if (!Bpacked) return 0;
    float* out_base = Bpacked;

    // We assume B is Fortran-order (K rows x N cols) with leading dim ldb
    for (int np = 0; np < n_panels; ++np) {
        int n0 = np * BLOCK_N;
        int nb = (N - n0) < BLOCK_N ? (N - n0) : BLOCK_N;

        for (int kp = 0; kp < k_panels; ++kp) {
            int k0 = kp * BLOCK_K;
            int kb = (K - k0) < BLOCK_K ? (K - k0) : BLOCK_K;

            // pack by micro-panels of MR rows:
            // for each micro-row-block r0 in [0, kb) step MR:
            for (int r0 = 0; r0 < kb; r0 += MR) {
                int mrem = kb - r0;
                int mr_here = mrem < MR ? mrem : MR;

                // For each column in this n-panel, copy MR elements from B[k0+r0 : k0+r0+mr_here, col]
                for (int col = 0; col < nb; ++col) {
                    int col_idx = n0 + col;
                    const float* src = B + (size_t)col_idx * (size_t)ldb + (size_t)(k0 + r0);

                    // copy mr_here entries
                    if (mr_here > 0) memcpy(out_base, src, (size_t)mr_here * sizeof(float));

                    // pad up to MR
                    if (mr_here < MR) {
                        memset(out_base + mr_here, 0, (size_t)(MR - mr_here) * sizeof(float));
                    }
                    out_base += MR;
                }
                // If nb < BLOCK_N, pad the remainder columns for this micro-panel
                if (nb < BLOCK_N) {
                    size_t pad_cols = (size_t)(BLOCK_N - nb);
                    memset(out_base, 0, pad_cols * MR * sizeof(float));
                    out_base += pad_cols * MR;
                }
            } // r0 loop
        } // kp loop
    } // np loop

    // Call the packed dispatcher
    cached_fn((const void*)A, (const void*)Bpacked, (void*)C,
              (size_t)M, (size_t)N, (size_t)K,
              (size_t)lda, (size_t)ldb, (size_t)ldc);

    free(Bpacked);
    return 1;
}

#ifdef __cplusplus
}
#endif

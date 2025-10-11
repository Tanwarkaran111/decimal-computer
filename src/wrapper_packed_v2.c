// wrapper_packed_v2.c
// Two packing variants (NK and KN) to test which packed-B layout the DLL expects.
// Build: gcc -shared -O3 -fPIC -I src src/wrapper_packed_v2.c -o build/gemm_native_packed_v2.dll -Wl,--export-all-symbols

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

// Change this to point at the packed-capable DLL you prefer
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

/* Block sizes observed in your DLL debug lines. They can be left as-is;
   kernels usually only require BLOCK_K to match micro-kernel depth. */
#define BLOCK_K 128
#define BLOCK_N 256

// Helper: allocate packed buffer (size K*N floats, padded to panel sizes)
static float* alloc_packed_buffer(int K, int N) {
    size_t total = (size_t)K * (size_t)N;
    float* buf = (float*) malloc(total * sizeof(float));
    return buf;
}

/* Variant 1: N-panels outer, then K-panels (np outer, kp inner)
   For each n-panel, for each k-panel, copy the kb x nb block column-by-column,
   storing each column as kb values padded to BLOCK_K. This matches many
   packed-B conventions where panel storage groups columns of B within each n-panel. */
__declspec(dllexport) int gemm_native_packed_pack_then_run_nk(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return 0;
    int n_panels = (N + BLOCK_N - 1) / BLOCK_N;
    int k_panels = (K + BLOCK_K - 1) / BLOCK_K;

    float* Bpacked = alloc_packed_buffer(K, N);
    if (!Bpacked) return 0;
    float* out = Bpacked;

    // Assume B is Fortran-order (column-major) with leading dim ldb (rows = K)
    for (int np = 0; np < n_panels; ++np) {
        int n0 = np * BLOCK_N;
        int nb = (N - n0) < BLOCK_N ? (N - n0) : BLOCK_N;
        for (int kp = 0; kp < k_panels; ++kp) {
            int k0 = kp * BLOCK_K;
            int kb = (K - k0) < BLOCK_K ? (K - k0) : BLOCK_K;
            // For each column in the nb block
            for (int col = 0; col < nb; ++col) {
                int col_idx = n0 + col;
                const float* src_col = B + (size_t)col_idx * (size_t)ldb + (size_t)k0;
                // copy kb elements
                if (kb > 0) memcpy(out, src_col, (size_t)kb * sizeof(float));
                // if kb < BLOCK_K pad the rest
                if (kb < BLOCK_K) memset(out + kb, 0, (size_t)(BLOCK_K - kb) * sizeof(float));
                out += BLOCK_K;
            }
            // pad remaining columns in panel to BLOCK_N
            if (nb < BLOCK_N) {
                size_t pad_cols = (size_t)(BLOCK_N - nb);
                memset(out, 0, pad_cols * (size_t)BLOCK_K * sizeof(float));
                out += pad_cols * BLOCK_K;
            }
        }
    }

    // call packed dispatcher
    cached_fn((const void*)A, (const void*)Bpacked, (void*)C,
              (size_t)M, (size_t)N, (size_t)K,
              (size_t)lda, (size_t)ldb, (size_t)ldc);

    free(Bpacked);
    return 1;
}

/* Variant 2: K-panels outer, then N-panels inner (kp outer, np inner)
   This stores data by k-block first (so panels group depth blocks before moving across N).
   Some kernels expect the packed buffer arranged primarily by K-blocks. */
__declspec(dllexport) int gemm_native_packed_pack_then_run_kn(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) return 0;
    int n_panels = (N + BLOCK_N - 1) / BLOCK_N;
    int k_panels = (K + BLOCK_K - 1) / BLOCK_K;

    float* Bpacked = alloc_packed_buffer(K, N);
    if (!Bpacked) return 0;
    float* out = Bpacked;

    // Assume B is Fortran-order (column-major) with leading dim ldb (rows = K)
    for (int kp = 0; kp < k_panels; ++kp) {
        int k0 = kp * BLOCK_K;
        int kb = (K - k0) < BLOCK_K ? (K - k0) : BLOCK_K;
        for (int np = 0; np < n_panels; ++np) {
            int n0 = np * BLOCK_N;
            int nb = (N - n0) < BLOCK_N ? (N - n0) : BLOCK_N;
            for (int col = 0; col < nb; ++col) {
                int col_idx = n0 + col;
                const float* src_col = B + (size_t)col_idx * (size_t)ldb + (size_t)k0;
                // copy kb elements
                if (kb > 0) memcpy(out, src_col, (size_t)kb * sizeof(float));
                // pad to BLOCK_K
                if (kb < BLOCK_K) memset(out + kb, 0, (size_t)(BLOCK_K - kb) * sizeof(float));
                out += BLOCK_K;
            }
            if (nb < BLOCK_N) {
                size_t pad_cols = (size_t)(BLOCK_N - nb);
                memset(out, 0, pad_cols * (size_t)BLOCK_K * sizeof(float));
                out += pad_cols * BLOCK_K;
            }
        }
    }

    // call packed dispatcher
    cached_fn((const void*)A, (const void*)Bpacked, (void*)C,
              (size_t)M, (size_t)N, (size_t)K,
              (size_t)lda, (size_t)ldb, (size_t)ldc);

    free(Bpacked);
    return 1;
}

#ifdef __cplusplus
}
#endif

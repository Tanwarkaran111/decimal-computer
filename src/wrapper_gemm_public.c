// wrapper_gemm_public.c
// Provides a clean public GEMM API that wraps the internal packed dispatcher.
//
// Build:
// gcc -shared -O3 -fPIC -I src src/wrapper_gemm_public.c -o build/gemm_native_public.dll -Wl,--export-all-symbols

#include <windows.h>
#include <malloc.h> 
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// === internal function signatures (from kernel_wrapper_packed.c) ===
typedef void (*pack_A_block_padded_t)(const double*, int, int, int, int, double*);
typedef void (*pack_B_block_padded_t)(const double*, int, int, int, int, double*, size_t);
typedef void (*gemm_dispatch_packed_t)(const double*, const double*, double*,
                                       size_t, size_t, size_t,
                                       size_t, size_t, size_t);

static const char *TARGET_DLL = "D:\\Invented_library\\build\\gemm_native_packed_stable320.dll";
static HMODULE h = NULL;
static pack_A_block_padded_t packA = NULL;
static pack_B_block_padded_t packB = NULL;
static gemm_dispatch_packed_t gemmPacked = NULL;

static inline int round_up_int(int x, int r) {
    return ((x + r - 1) / r) * r;
}

static int ensure_loaded(void) {
    if (h) return 1;
    h = LoadLibraryA(TARGET_DLL);
    if (!h) return 0;
    packA = (pack_A_block_padded_t)GetProcAddress(h, "pack_A_block_padded");
    packB = (pack_B_block_padded_t)GetProcAddress(h, "pack_B_block_padded");
    gemmPacked = (gemm_dispatch_packed_t)GetProcAddress(h, "gemm_dispatch_packed");
    return (packA && packB && gemmPacked);
}

// Tile sizes taken from your kernel
#define MIC_M_TILE 8
#define MIC_N_TILE 8
#define MIC_K_TILE 4

__declspec(dllexport)
int gemm_native_public(const double *A, const double *B, double *C,
                       int M, int N, int K)
{
    if (!ensure_loaded()) return 0;

    int M_pad = round_up_int(M, MIC_M_TILE);
    int N_pad = round_up_int(N, MIC_N_TILE);
    int K_pad = round_up_int(K, MIC_K_TILE);

    // allocate padded copies
    double *A_pack = (double*)_aligned_malloc((size_t)M_pad * K_pad * sizeof(double), 64);
    double *B_pack = (double*)_aligned_malloc((size_t)K_pad * N_pad * sizeof(double), 64);
    if (!A_pack || !B_pack) {
        if (A_pack) _aligned_free(A_pack);
        if (B_pack) _aligned_free(B_pack);
        return 0;
    }

    // call internal packers
    packA(A, K, M, K, K_pad, A_pack);
    packB(B, K, K, K_pad, N, B_pack, (size_t)K_pad * N_pad);

    // dispatch packed GEMM
    gemmPacked(A_pack, B_pack, C, M, N, K, K, K, M);

    _aligned_free(A_pack);
    _aligned_free(B_pack);
    return 1;
}

#ifdef __cplusplus
}
#endif

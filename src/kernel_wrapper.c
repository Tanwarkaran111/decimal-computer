// src/kernel_wrapper.c
// Thin wrapper that tiles a large MxN GEMM and calls a micro-kernel (AVX2 or scalar).
// Matches calling convention used earlier files:
//   A: row-major M x K -> A[i*lda + k]
//   B: row-major K x N -> B[k*ldb + j]
//   C: row-major M x N -> C[i*ldc + j]
// The function performs C += A * B.

#include "../include/micro_kernel.h"
#include <stddef.h>
#include <stdint.h>

#ifndef BLOCK_M
#define BLOCK_M 120
#endif
#ifndef BLOCK_N
#define BLOCK_N 256
#endif
#ifndef BLOCK_K
#define BLOCK_K 64
#endif

// gemm_dispatch: top-level tiling + dispatch
// use_avx2: 1 => call micro_kernel_avx2, 0 => call micro_kernel_scalar
void gemm_dispatch(const double *A, const double *B, double *C,
                   int M, int N, int K,
                   int lda, int ldb, int ldc,
                   int use_avx2)
{
    // Basic blocking loops
    for (int bm = 0; bm < M; bm += BLOCK_M) {
        int mlen = (M - bm) < BLOCK_M ? (M - bm) : BLOCK_M;
        for (int bn = 0; bn < N; bn += BLOCK_N) {
            int nlen = (N - bn) < BLOCK_N ? (N - bn) : BLOCK_N;
            for (int bk = 0; bk < K; bk += BLOCK_K) {
                int klen = (K - bk) < BLOCK_K ? (K - bk) : BLOCK_K;

                // Inner MR x NR tiling (MR=6, NR=8 expected by AVX2 kernels)
                for (int i = 0; i < mlen; i += 6) {
                    int M_tile = (mlen - i) < 6 ? (mlen - i) : 6;
                    for (int j = 0; j < nlen; j += 8) {
                        int N_tile = (nlen - j) < 8 ? (nlen - j) : 8;

                        const double *A_tile = A + (bm + i) * lda + bk;
                        const double *B_tile = B + bk * ldb + (bn + j);
                        double *C_tile = C + (bm + i) * ldc + (bn + j);

                        // Dispatch between scalar, baseline AVX2, and optimized AVX2 kernels
                        if (use_avx2 == 2) {
                            micro_kernel_avx2_opt(A_tile, B_tile, C_tile,
                                                  lda, ldb, ldc,
                                                  M_tile, N_tile, klen);
                        } else if (use_avx2 == 1) {
                            micro_kernel_avx2(A_tile, B_tile, C_tile,
                                              lda, ldb, ldc,
                                              M_tile, N_tile, klen);
                        } else {
                            micro_kernel_scalar(A_tile, B_tile, C_tile,
                                                lda, ldb, ldc,
                                                M_tile, N_tile, klen);
                        }
                    }
                }
            }
        }
    }
}

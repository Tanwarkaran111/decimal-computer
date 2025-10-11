#include "../include/micro_kernel.h"
#include <stddef.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BLOCK_M
#define BLOCK_M 256
#endif
#ifndef BLOCK_N
#define BLOCK_N 256
#endif
#ifndef BLOCK_K
#define BLOCK_K 256
#endif

/* MSVC-friendly gemm_dispatch: declare loop counters outside pragma,
   use #pragma omp parallel / #pragma omp for to avoid C3015 issues. */
void gemm_dispatch(const double *A, const double *B, double *C,
                   int M, int N, int K,
                   int lda, int ldb, int ldc,
                   int use_avx2)
{
    if (M <= 0 || N <= 0) return;

    /* compute tile counts as plain ints BEFORE the pragma */
    int tiles_m_i = (M + BLOCK_M - 1) / BLOCK_M;
    int tiles_n_i = (N + BLOCK_N - 1) / BLOCK_N;
    int total_tiles_i = tiles_m_i * tiles_n_i;

    /* declare loop counters outside pragma to satisfy MSVC OpenMP canonical form */
    int t;
#if defined(_OPENMP)
#pragma omp parallel
    {
        /* distribute the for-loop iterations among threads */
#pragma omp for schedule(dynamic)
        for (t = 0; t < total_tiles_i; ++t) {
            int tm = t / tiles_n_i;
            int tn = t % tiles_n_i;

            int bm = tm * BLOCK_M;
            int bn = tn * BLOCK_N;

            int mlen = (M - bm) < BLOCK_M ? (M - bm) : BLOCK_M;
            int nlen = (N - bn) < BLOCK_N ? (N - bn) : BLOCK_N;

            for (int bk = 0; bk < K; bk += BLOCK_K) {
                int klen = (K - bk) < BLOCK_K ? (K - bk) : BLOCK_K;

                for (int i = 0; i < mlen; i += 6) {
                    int M_tile = (mlen - i) < 6 ? (mlen - i) : 6;
                    for (int j = 0; j < nlen; j += 8) {
                        int N_tile = (nlen - j) < 8 ? (nlen - j) : 8;

                        const double *A_tile = A + (size_t)(bm + i) * (size_t)lda + (size_t)bk;
                        const double *B_tile = B + (size_t)bk * (size_t)ldb + (size_t)(bn + j);
                        double *C_tile = C + (size_t)(bm + i) * (size_t)ldc + (size_t)(bn + j);

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
        } /* end omp for */
    } /* end omp parallel */
#else
    for (int bm = 0; bm < M; bm += BLOCK_M) {
        int mlen = (M - bm) < BLOCK_M ? (M - bm) : BLOCK_M;
        for (int bn = 0; bn < N; bn += BLOCK_N) {
            int nlen = (N - bn) < BLOCK_N ? (N - bn) : BLOCK_N;
            for (int bk = 0; bk < K; bk += BLOCK_K) {
                int klen = (K - bk) < BLOCK_K ? (K - bk) : BLOCK_K;

                for (int i = 0; i < mlen; i += 6) {
                    int M_tile = (mlen - i) < 6 ? (mlen - i) : 6;
                    for (int j = 0; j < nlen; j += 8) {
                        int N_tile = (nlen - j) < 8 ? (nlen - j) : 8;

                        const double *A_tile = A + (size_t)(bm + i) * (size_t)lda + (size_t)bk;
                        const double *B_tile = B + (size_t)bk * (size_t)ldb + (size_t)(bn + j);
                        double *C_tile = C + (size_t)(bm + i) * (size_t)ldc + (size_t)(bn + j);

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
#endif
}

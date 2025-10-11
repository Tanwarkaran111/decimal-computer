// src/micro_kernel_avx2_opt.c
// Optimized AVX2+FMA micro-kernel MR=6 x NR=8 (double-precision).
// - K loop unrolled by 4
// - Uses 12 vector accumulators (6 rows x 2 vec blocks)
// - Prefetch B rows
// - Safe for partial tiles (M_tile <= 6, N_tile <= 8)
// Compile with -O3 -mavx2 -mfma

#include <immintrin.h>
#include <stddef.h>
#include "../include/micro_kernel.h"

void micro_kernel_avx2_opt(const double *A, const double *B, double *C,
                           int lda, int ldb, int ldc,
                           int M_tile, int N_tile, int K_tile)
{
    // Quick fallback for very small K to reuse previous AVX2 (or scalar)
    if (K_tile <= 2) {
        // fallback to basic AVX2 kernel (if you want), but for safety call scalar
        micro_kernel_scalar(A, B, C, lda, ldb, ldc, M_tile, N_tile, K_tile);
        return;
    }

    // Initialize accumulators: acc[row][vec_block]
    __m256d acc0_0 = _mm256_setzero_pd();
    __m256d acc0_1 = _mm256_setzero_pd();
    __m256d acc1_0 = _mm256_setzero_pd();
    __m256d acc1_1 = _mm256_setzero_pd();
    __m256d acc2_0 = _mm256_setzero_pd();
    __m256d acc2_1 = _mm256_setzero_pd();
    __m256d acc3_0 = _mm256_setzero_pd();
    __m256d acc3_1 = _mm256_setzero_pd();
    __m256d acc4_0 = _mm256_setzero_pd();
    __m256d acc4_1 = _mm256_setzero_pd();
    __m256d acc5_0 = _mm256_setzero_pd();
    __m256d acc5_1 = _mm256_setzero_pd();

    int k = 0;

    // Main unrolled loop: process 4 K at a time
    for (; k + 3 < K_tile; k += 4) {
        // Prefetch a future B row (heuristic)
        int pf_k = k + 8;
        if (pf_k < K_tile) {
            _mm_prefetch((const char*)(B + pf_k * ldb), _MM_HINT_T0);
        }

        // process k, k+1, k+2, k+3
        for (int ku = 0; ku < 4; ++ku) {
            const double *B_row = B + (k + ku) * ldb;

            // Load B vectors (cols 0..3 and 4..7) if present
            __m256d b0 = _mm256_setzero_pd();
            __m256d b1 = _mm256_setzero_pd();
            if (N_tile >= 4) b0 = _mm256_loadu_pd(B_row + 0);
            if (N_tile >= 8) b1 = _mm256_loadu_pd(B_row + 4);

            // For each row i, broadcast A[i,k+ku] and FMA
            // Row 0
            if (M_tile > 0) {
                const double a0 = A[0 * lda + (k + ku)];
                __m256d aa0 = _mm256_broadcast_sd(&a0);
                if (N_tile >= 4) acc0_0 = _mm256_fmadd_pd(b0, aa0, acc0_0);
                if (N_tile >= 8) acc0_1 = _mm256_fmadd_pd(b1, aa0, acc0_1);
            }
            // Row 1
            if (M_tile > 1) {
                const double a1 = A[1 * lda + (k + ku)];
                __m256d aa1 = _mm256_broadcast_sd(&a1);
                if (N_tile >= 4) acc1_0 = _mm256_fmadd_pd(b0, aa1, acc1_0);
                if (N_tile >= 8) acc1_1 = _mm256_fmadd_pd(b1, aa1, acc1_1);
            }
            // Row 2
            if (M_tile > 2) {
                const double a2 = A[2 * lda + (k + ku)];
                __m256d aa2 = _mm256_broadcast_sd(&a2);
                if (N_tile >= 4) acc2_0 = _mm256_fmadd_pd(b0, aa2, acc2_0);
                if (N_tile >= 8) acc2_1 = _mm256_fmadd_pd(b1, aa2, acc2_1);
            }
            // Row 3
            if (M_tile > 3) {
                const double a3 = A[3 * lda + (k + ku)];
                __m256d aa3 = _mm256_broadcast_sd(&a3);
                if (N_tile >= 4) acc3_0 = _mm256_fmadd_pd(b0, aa3, acc3_0);
                if (N_tile >= 8) acc3_1 = _mm256_fmadd_pd(b1, aa3, acc3_1);
            }
            // Row 4
            if (M_tile > 4) {
                const double a4 = A[4 * lda + (k + ku)];
                __m256d aa4 = _mm256_broadcast_sd(&a4);
                if (N_tile >= 4) acc4_0 = _mm256_fmadd_pd(b0, aa4, acc4_0);
                if (N_tile >= 8) acc4_1 = _mm256_fmadd_pd(b1, aa4, acc4_1);
            }
            // Row 5
            if (M_tile > 5) {
                const double a5 = A[5 * lda + (k + ku)];
                __m256d aa5 = _mm256_broadcast_sd(&a5);
                if (N_tile >= 4) acc5_0 = _mm256_fmadd_pd(b0, aa5, acc5_0);
                if (N_tile >= 8) acc5_1 = _mm256_fmadd_pd(b1, aa5, acc5_1);
            }
        }
    }

    // Remainder K (k .. K_tile-1)
    for (; k < K_tile; ++k) {
        const double *B_row = B + k * ldb;
        __m256d b0 = _mm256_setzero_pd();
        __m256d b1 = _mm256_setzero_pd();
        if (N_tile >= 4) b0 = _mm256_loadu_pd(B_row + 0);
        if (N_tile >= 8) b1 = _mm256_loadu_pd(B_row + 4);

        if (M_tile > 0) {
            const double a0 = A[0 * lda + k];
            __m256d aa0 = _mm256_broadcast_sd(&a0);
            if (N_tile >= 4) acc0_0 = _mm256_fmadd_pd(b0, aa0, acc0_0);
            if (N_tile >= 8) acc0_1 = _mm256_fmadd_pd(b1, aa0, acc0_1);
        }
        if (M_tile > 1) {
            const double a1 = A[1 * lda + k];
            __m256d aa1 = _mm256_broadcast_sd(&a1);
            if (N_tile >= 4) acc1_0 = _mm256_fmadd_pd(b0, aa1, acc1_0);
            if (N_tile >= 8) acc1_1 = _mm256_fmadd_pd(b1, aa1, acc1_1);
        }
        if (M_tile > 2) {
            const double a2 = A[2 * lda + k];
            __m256d aa2 = _mm256_broadcast_sd(&a2);
            if (N_tile >= 4) acc2_0 = _mm256_fmadd_pd(b0, aa2, acc2_0);
            if (N_tile >= 8) acc2_1 = _mm256_fmadd_pd(b1, aa2, acc2_1);
        }
        if (M_tile > 3) {
            const double a3 = A[3 * lda + k];
            __m256d aa3 = _mm256_broadcast_sd(&a3);
            if (N_tile >= 4) acc3_0 = _mm256_fmadd_pd(b0, aa3, acc3_0);
            if (N_tile >= 8) acc3_1 = _mm256_fmadd_pd(b1, aa3, acc3_1);
        }
        if (M_tile > 4) {
            const double a4 = A[4 * lda + k];
            __m256d aa4 = _mm256_broadcast_sd(&a4);
            if (N_tile >= 4) acc4_0 = _mm256_fmadd_pd(b0, aa4, acc4_0);
            if (N_tile >= 8) acc4_1 = _mm256_fmadd_pd(b1, aa4, acc4_1);
        }
        if (M_tile > 5) {
            const double a5 = A[5 * lda + k];
            __m256d aa5 = _mm256_broadcast_sd(&a5);
            if (N_tile >= 4) acc5_0 = _mm256_fmadd_pd(b0, aa5, acc5_0);
            if (N_tile >= 8) acc5_1 = _mm256_fmadd_pd(b1, aa5, acc5_1);
        }
    }

    // Store vector accumulators back into C
    double tmp[4];
    // columns 0..3
    if (N_tile >= 4) {
        if (M_tile > 0) { _mm256_storeu_pd(tmp, acc0_0); for (int j=0;j<4 && j<N_tile;j++) C[0*ldc + j] += tmp[j]; }
        if (M_tile > 1) { _mm256_storeu_pd(tmp, acc1_0); for (int j=0;j<4 && j<N_tile;j++) C[1*ldc + j] += tmp[j]; }
        if (M_tile > 2) { _mm256_storeu_pd(tmp, acc2_0); for (int j=0;j<4 && j<N_tile;j++) C[2*ldc + j] += tmp[j]; }
        if (M_tile > 3) { _mm256_storeu_pd(tmp, acc3_0); for (int j=0;j<4 && j<N_tile;j++) C[3*ldc + j] += tmp[j]; }
        if (M_tile > 4) { _mm256_storeu_pd(tmp, acc4_0); for (int j=0;j<4 && j<N_tile;j++) C[4*ldc + j] += tmp[j]; }
        if (M_tile > 5) { _mm256_storeu_pd(tmp, acc5_0); for (int j=0;j<4 && j<N_tile;j++) C[5*ldc + j] += tmp[j]; }
    }

    // columns 4..7
    if (N_tile >= 8) {
        if (M_tile > 0) { _mm256_storeu_pd(tmp, acc0_1); for (int jj=0;jj<4;jj++) C[0*ldc + 4 + jj] += tmp[jj]; }
        if (M_tile > 1) { _mm256_storeu_pd(tmp, acc1_1); for (int jj=0;jj<4;jj++) C[1*ldc + 4 + jj] += tmp[jj]; }
        if (M_tile > 2) { _mm256_storeu_pd(tmp, acc2_1); for (int jj=0;jj<4;jj++) C[2*ldc + 4 + jj] += tmp[jj]; }
        if (M_tile > 3) { _mm256_storeu_pd(tmp, acc3_1); for (int jj=0;jj<4;jj++) C[3*ldc + 4 + jj] += tmp[jj]; }
        if (M_tile > 4) { _mm256_storeu_pd(tmp, acc4_1); for (int jj=0;jj<4;jj++) C[4*ldc + 4 + jj] += tmp[jj]; }
        if (M_tile > 5) { _mm256_storeu_pd(tmp, acc5_1); for (int jj=0;jj<4;jj++) C[5*ldc + 4 + jj] += tmp[jj]; }
    }

    // Scalar tail handling for columns < 4 or 4<=N_tile<8
    // We handle any remaining scalar columns (j from scalar_col_start to N_tile-1)
    int scalar_col_start = 0;
    if (N_tile >= 8) scalar_col_start = 8;
    else if (N_tile >= 4) scalar_col_start = 4;
    else scalar_col_start = 0;

    if (scalar_col_start < N_tile) {
        double scalar_sum[6][8];
        for (int i = 0; i < M_tile; ++i)
            for (int j = scalar_col_start; j < N_tile; ++j)
                scalar_sum[i][j - scalar_col_start] = 0.0;

        for (int kk = 0; kk < K_tile; ++kk) {
            const double *B_row = B + kk * ldb;
            for (int j = scalar_col_start; j < N_tile; ++j) {
                double bj = B_row[j];
                for (int i = 0; i < M_tile; ++i) {
                    double a = A[i * lda + kk];
                    scalar_sum[i][j - scalar_col_start] += a * bj;
                }
            }
        }
        for (int i = 0; i < M_tile; ++i) {
            double *Crow = C + i * ldc;
            for (int j = scalar_col_start; j < N_tile; ++j) {
                Crow[j] += scalar_sum[i][j - scalar_col_start];
            }
        }
    }
}

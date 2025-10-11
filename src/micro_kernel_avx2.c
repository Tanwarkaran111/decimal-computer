// src/micro_kernel_avx2.c
// AVX2+FMA micro-kernel MR=6 x NR=8 (double-precision)
//
// Assumptions / calling convention:
// - Row-major layout (C style).
//   A: M x K, accessed as A[i*lda + k]
//   B: K x N, accessed as B[k*ldb + j]
//   C: M x N, accessed as C[i*ldc + j]
// - lda = leading dimension (stride) of A in elements (usually K)
// - ldb = leading dimension of B in elements (usually N)
// - ldc = leading dimension of C in elements (usually N)
// - M_tile <= 6, N_tile <= 8, K_tile >= 1 (K tile is variable).
// - The kernel *accumulates* into C (i.e., it does C += A*B for the tile).
//
// Notes:
// - This file uses AVX2 intrinsics and FMA (_mm256_fmadd_pd). Compile with:
//     gcc -O3 -mavx2 -mfma -std=c11 ...
// - For best speed, feed well aligned contiguous arrays (np.ascontiguousarray with dtype=np.float64).
// - We prefetch parts of B (and A) for future K iterations (simple strategy).
//
// The kernel vectorizes across the N dimension (columns) using __m256d (4 doubles).
// We keep two vector blocks per row: cols 0..3 and cols 4..7. If N_tile < 4/8,
// scalar code covers the remainder.

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

void micro_kernel_avx2(const double *A, const double *B, double *C,
                       int lda, int ldb, int ldc,
                       int M_tile, int N_tile, int K_tile)
{
    // Zero accumulators:
    // For each row i (0..5) we keep two __m256d accumulators (cols 0-3, 4-7).
    __m256d acc[6][2];

    for (int i = 0; i < 6; ++i) {
        acc[i][0] = _mm256_setzero_pd();
        acc[i][1] = _mm256_setzero_pd();
    }

    // Main K loop: vectorized multiply-add into accumulators.
    for (int k = 0; k < K_tile; ++k) {
        // Prefetch future B rows to L1 (simple heuristic)
        // Prefetch B row at k+2 if exists.
        const int pf_k = k + 2;
        if (pf_k < K_tile) {
            // Prefetch first 64 bytes of B[pf_k]
            _mm_prefetch((const char*)(B + pf_k * ldb), _MM_HINT_T0);
        }

        // Pointer to B row k
        const double *B_row = B + k * ldb;

        // Load vector chunks of B row where available
        int have_vec0 = (N_tile >= 4);
        int have_vec1 = (N_tile >= 8);

        __m256d bvec0 = _mm256_setzero_pd(); // columns 0..3
        __m256d bvec1 = _mm256_setzero_pd(); // columns 4..7

        if (have_vec0) {
            bvec0 = _mm256_loadu_pd(B_row + 0); // safe even if unaligned
        }
        if (have_vec1) {
            bvec1 = _mm256_loadu_pd(B_row + 4);
        }

        // For each row i we broadcast A[i,k] and do FMA: acc += bvec * a_bcast
        for (int i = 0; i < M_tile; ++i) {
            // A_ik scalar
            const double a_val = A[i * lda + k];
            __m256d a_bcast = _mm256_broadcast_sd(&a_val);
            if (have_vec0) {
                acc[i][0] = _mm256_fmadd_pd(bvec0, a_bcast, acc[i][0]);
            } else {
                // No vector block 0 (N_tile < 4): fallback handled below
            }
            if (have_vec1) {
                acc[i][1] = _mm256_fmadd_pd(bvec1, a_bcast, acc[i][1]);
            }
        }

        // For columns that are not covered by vector blocks (N_tile % 4 and tails),
        // we do scalar accumulation per remaining column.
        int n_scalar_start = (N_tile >= 8) ? 8 : ((N_tile >= 4) ? 4 : 0);
        for (int j = n_scalar_start; j < N_tile; ++j) {
            // scalar B[k,j]
            double bj = B_row[j];
            for (int i = 0; i < M_tile; ++i) {
                double a_val = A[i * lda + k];
                // We accumulate into C directly into a small temp variable in place of separate scalar accumulators:
                // But to keep numerically consistent with vector path, we accumulate into C via temporary scalar sums.
                // We'll store these scalar sums after finishing K loop (we need per-row scalar accumulators).
                // To simplify: allocate on stack small temporary accumulators for scalar tail columns.
                // However, to avoid dynamic allocation per k, we'll implement scalar tail accumulation after K loop.
                // So nothing here for scalar tails (we'll compute scalar tails below).
            }
        }
        // Note: scalar-tail approach implemented after finishing entire K loop.
    } // end K loop

    // Write back vectorized accumulators to C (cols 0..3 and 4..7)
    // We'll write vector parts first, then handle scalar tail columns (if any) with a separate scalar multiply-accumulate.

    // Temporary storage for extracting vector accumulators
    double tmp[4];

    // Vector blocks storage
    if (N_tile >= 4) {
        // store cols 0..3
        for (int i = 0; i < M_tile; ++i) {
            _mm256_storeu_pd(tmp, acc[i][0]); // tmp[0..3] correspond to columns j=0..3
            // C row pointer
            double *Crow = C + i * ldc;
            // Accumulate into C
            for (int j = 0; j < 4 && j < N_tile; ++j) {
                Crow[j] += tmp[j];
            }
        }
    }
    if (N_tile >= 8) {
        // store cols 4..7
        for (int i = 0; i < M_tile; ++i) {
            _mm256_storeu_pd(tmp, acc[i][1]); // tmp[0..3] correspond to columns j=4..7
            double *Crow = C + i * ldc;
            for (int jj = 0; jj < 4; ++jj) {
                int j = 4 + jj;
                if (j < N_tile) Crow[j] += tmp[jj];
            }
        }
    }

    // --- Scalar tail handling for columns not covered by vector blocks ---
    // This handles columns j in [0,N_tile) not handled above (i.e., when N_tile < 4, or N_tile%4 != 0)
    // We'll compute scalar accumulators sum_scalar[i][j] = sum_k A[i,k] * B[k,j] for the remaining columns.
    int scalar_col_start = 0;
    if (N_tile >= 8) scalar_col_start = 8;
    else if (N_tile >= 4) scalar_col_start = 4;
    else scalar_col_start = 0;

    if (scalar_col_start < N_tile) {
        // small stack buffer for scalar results: up to 8 columns and up to 6 rows -> safe on stack
        double scalar_sum[6][8]; // zero-initialize
        for (int i = 0; i < M_tile; ++i) {
            for (int j = scalar_col_start; j < N_tile; ++j) {
                scalar_sum[i][j - scalar_col_start] = 0.0;
            }
        }

        // Compute scalar sums across K
        for (int k = 0; k < K_tile; ++k) {
            const double *B_row = B + k * ldb;
            for (int j = scalar_col_start; j < N_tile; ++j) {
                double bj = B_row[j];
                for (int i = 0; i < M_tile; ++i) {
                    double a = A[i * lda + k];
                    scalar_sum[i][j - scalar_col_start] += a * bj;
                }
            }
        }

        // Add scalar sums into C
        for (int i = 0; i < M_tile; ++i) {
            double *Crow = C + i * ldc;
            for (int j = scalar_col_start; j < N_tile; ++j) {
                Crow[j] += scalar_sum[i][j - scalar_col_start];
            }
        }
    }
}

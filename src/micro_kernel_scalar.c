// src/micro_kernel_scalar.c
// Scalar fallback micro-kernel: correct and safe, used when AVX2/FMA is not available.
// Keeps the same calling convention as the AVX2 kernel in your repo.
//
// Compile with any C compiler (no special flags required).

#include "../include/micro_kernel.h"

void micro_kernel_scalar(const double *A, const double *B, double *C,
                         int lda, int ldb, int ldc,
                         int M_tile, int N_tile, int K_tile)
{
    // For each row in A tile
    for (int i = 0; i < M_tile; ++i) {
        // pointer to row i of C (tile)
        double *Crow = C + i * ldc;
        // We'll compute sums for columns j in [0, N_tile)
        for (int j = 0; j < N_tile; ++j) {
            double sum = 0.0;
            // accumulate over K
            for (int k = 0; k < K_tile; ++k) {
                double a = A[i * lda + k];
                double b = B[k * ldb + j];
                sum += a * b;
            }
            Crow[j] += sum;
        }
    }
}

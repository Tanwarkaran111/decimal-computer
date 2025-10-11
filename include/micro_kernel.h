// include/micro_kernel.h
#ifndef MICRO_KERNEL_H
#define MICRO_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

void gemm_dispatch_packed(const void *A, const void *B, void *C,
                          size_t M, size_t N, size_t K,
                          size_t lda, size_t ldb, size_t ldc);


void micro_kernel_avx2(const double *A, const double *B, double *C,
                       int lda, int ldb, int ldc,
                       int M_tile, int N_tile, int K_tile);

void micro_kernel_avx2_opt(const double *A, const double *B, double *C,
                           int lda, int ldb, int ldc,
                           int M_tile, int N_tile, int K_tile);

void micro_kernel_scalar(const double *A, const double *B, double *C,
                         int lda, int ldb, int ldc,
                         int M_tile, int N_tile, int K_tile);

#ifdef __cplusplus
}
#endif

#endif // MICRO_KERNEL_H

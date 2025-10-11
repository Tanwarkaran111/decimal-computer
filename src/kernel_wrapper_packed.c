// src/kernel_wrapper_packed.c
// Very defensive version: large padding, zeroed buffers, strict bounds checks.

#include "../include/micro_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>

#ifdef _WIN32
#include <malloc.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifndef BLOCK_M
#define BLOCK_M 192
#endif
#ifndef BLOCK_N
#define BLOCK_N 256
#endif
#ifndef BLOCK_K
#define BLOCK_K 128
#endif

/* micro-kernel tile widths (must match your micro-kernel) */
#ifndef MIC_N_TILE
#define MIC_N_TILE 8
#endif
#ifndef MIC_K_TILE
#define MIC_K_TILE 4
#endif

/* HUGE safety pad appended to each per-thread buffer (1 MiB). */
#ifndef GEMM_SAFE_PAD_BYTES
#define GEMM_SAFE_PAD_BYTES (1 << 20)
#endif

static int env_or_default(const char *name, int def) {
    const char *v = getenv(name);
    if (!v) return def;
    int x = atoi(v);
    return x > 0 ? x : def;
}

static inline void *aligned_alloc_local(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    return p;
#endif
}
static inline void aligned_free_local(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* round up integer to multiple */
static inline int round_up_int(int x, int r) {
    if (r <= 0) return x;
    return ((x + r - 1) / r) * r;
}

/* Pack B block into packed buffer, klen_padded stride per column. Zero tail up to klen_padded. */
static void pack_B_block_padded(const double *B, int ldb,
                                int K_tile, int klen_padded, int N_tile,
                                double *B_packed, size_t per_buf_elems)
{
    /* Safety: ensure caller ensured klen_padded * N_tile <= per_buf_elems */
    for (int j = 0; j < N_tile; ++j) {
        double *dst = B_packed + (size_t)j * (size_t)klen_padded;
        const double *src_col = B + (size_t)j;
        for (int k = 0; k < K_tile; ++k) {
            dst[k] = src_col[(size_t)k * (size_t)ldb];
        }
        for (int k = K_tile; k < klen_padded; ++k) {
            dst[k] = 0.0;
        }
    }
}

/* Exports */
EXPORT void gemm_dispatch_packed(
    const void *A_v, const void *B_v, void *C_v,
    size_t M_s, size_t N_s, size_t K_s,
    size_t lda_s, size_t ldb_s, size_t ldc_s
);

void gemm_dispatch_packed_impl(
    const double *A, const double *B, double *C,
    int M, int N, int K,
    int lda, int ldb, int ldc
);

EXPORT void gemm_dispatch_packed(
    const void *A_v, const void *B_v, void *C_v,
    size_t M_s, size_t N_s, size_t K_s,
    size_t lda_s, size_t ldb_s, size_t ldc_s
) {
    if (!A_v || !B_v || !C_v) {
        fprintf(stderr, "gemm_dispatch_packed: NULL pointer A=%p B=%p C=%p\n", A_v, B_v, C_v);
        fflush(stderr);
        return;
    }
    if (M_s == 0 || N_s == 0 || K_s == 0) {
        fprintf(stderr, "gemm_dispatch_packed: zero-dimension M=%zu N=%zu K=%zu\n", M_s, N_s, K_s);
        fflush(stderr);
        return;
    }
    if (M_s > INT32_MAX || N_s > INT32_MAX || K_s > INT32_MAX ||
        lda_s > INT32_MAX || ldb_s > INT32_MAX || ldc_s > INT32_MAX) {
        fprintf(stderr, "gemm_dispatch_packed: dims exceed 32-bit\n");
        fflush(stderr);
        return;
    }

    gemm_dispatch_packed_impl((const double*)A_v, (const double*)B_v, (double*)C_v,
                              (int)M_s, (int)N_s, (int)K_s,
                              (int)lda_s, (int)ldb_s, (int)ldc_s);
}

void gemm_dispatch_packed_impl(const double *A, const double *B, double *C,
                          int M, int N, int K,
                          int lda, int ldb, int ldc)
{
    int BLK_M = env_or_default("GEMM_BLOCK_M", BLOCK_M);
    int BLK_N = env_or_default("GEMM_BLOCK_N", BLOCK_N);
    int BLK_K = env_or_default("GEMM_BLOCK_K", BLOCK_K);

    fprintf(stderr, "DBG entry: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d BLK_M=%d BLK_N=%d BLK_K=%d\n",
            M,N,K,lda,ldb,ldc,BLK_M,BLK_N,BLK_K);
    fflush(stderr);

    int nprocs = omp_get_num_procs();
    int nthreads = omp_get_max_threads();

    int BLK_K_padded = round_up_int(BLK_K, MIC_K_TILE);
    size_t extra_cols = (MIC_N_TILE > 0) ? (MIC_N_TILE - 1) : 0;

    /* allocate per-thread buffer size based on BLK_K_padded and BLK_N + extra_cols */
    size_t per_buf_elems = (size_t)BLK_K_padded * ((size_t)BLK_N + extra_cols);
    size_t per_buf_bytes = per_buf_elems * sizeof(double);
    size_t pad_bytes = (size_t)GEMM_SAFE_PAD_BYTES;
    size_t alloc_bytes = per_buf_bytes + pad_bytes;

    fprintf(stderr, "DBG buffers: nprocs=%d nthreads=%d per_buf_bytes=%zu pad_bytes=%zu alloc_bytes=%zu\n",
            nprocs, nthreads, per_buf_bytes, pad_bytes, alloc_bytes);
    fflush(stderr);

    double *single_buffer = NULL;
    double **B_packed_arr = NULL;

    /* prefer single buffer in single-threaded runs (simpler) */
    if (nthreads <= 1) {
        single_buffer = (double*) aligned_alloc_local(64, alloc_bytes);
        if (!single_buffer) {
            fprintf(stderr, "ERROR: single_buffer alloc failed (%zu bytes)\n", alloc_bytes);
            fflush(stderr);
            return;
        }
        /* zero the whole area to make accidental reads safe */
        memset(single_buffer, 0, alloc_bytes);
        fprintf(stderr, "DBG single-buffer allocated %p\n", (void*)single_buffer);
        fflush(stderr);
    } else {
        B_packed_arr = (double**) malloc(sizeof(double*) * (size_t)nthreads);
        if (!B_packed_arr) {
            fprintf(stderr, "ERROR: B_packed_arr alloc failed\n");
            fflush(stderr);
            return;
        }
        for (int t = 0; t < nthreads; ++t) {
            B_packed_arr[t] = (double*) aligned_alloc_local(64, alloc_bytes);
            if (B_packed_arr[t]) {
                memset(B_packed_arr[t], 0, alloc_bytes);
            } else {
                fprintf(stderr, "WARN: per-thread alloc failed for t=%d\n", t);
                fflush(stderr);
            }
        }
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *B_packed = (nthreads <= 1) ? single_buffer : B_packed_arr[tid];

        if (!B_packed) {
            #pragma omp barrier
        } else {
            #pragma omp for schedule(static)
            for (int bm = 0; bm < M; bm += BLK_M) {
                int mlen = (M - bm < BLK_M) ? (M - bm) : BLK_M;
                for (int bk = 0; bk < K; bk += BLK_K) {
                    int klen = (K - bk < BLK_K) ? (K - bk) : BLK_K;
                    int klen_padded = round_up_int(klen, MIC_K_TILE);
                    for (int bn = 0; bn < N; bn += BLK_N) {
                        int nlen = (N - bn < BLK_N) ? (N - bn) : BLK_N;

                        const double *B_block_src = B + (size_t)bk * (size_t)ldb + (size_t)bn;

                        fprintf(stderr, "DBG tile tid=%d bm=%d bk=%d bn=%d mlen=%d klen=%d klen_padded=%d nlen=%d B_block_src=%p B_packed=%p\n",
                                tid, bm, bk, bn, mlen, klen, klen_padded, nlen,
                                (const void*)B_block_src, (void*)B_packed);
                        fflush(stderr);

                        /* bounds: ensure packing fits into per_buf_elems */
                        size_t needed_elems = (size_t)klen_padded * (size_t)nlen;
                        if (needed_elems > per_buf_elems) {
                            fprintf(stderr, "FATAL PACK OVERFLOW: tid=%d klen_padded=%d nlen=%d needed_elems=%zu per_buf_elems=%zu\n",
                                    tid, klen_padded, nlen, needed_elems, per_buf_elems);
                            fflush(stderr);
                            /* skip this block to avoid crash */
                            continue;
                        }

                        /* pack */
                        pack_B_block_padded(B_block_src, ldb, klen, klen_padded, nlen, B_packed, per_buf_elems);

                        for (int i = 0; i < mlen; i += 6) {
                            int M_tile = (mlen - i < 6) ? (mlen - i) : 6;
                            for (int j = 0; j < nlen; j += MIC_N_TILE) {
                                int N_tile = (nlen - j < MIC_N_TILE) ? (nlen - j) : MIC_N_TILE;

                                const double *A_tile = A + (size_t)(bm + i) * (size_t)lda + (size_t)bk;
                                const double *B_tile = B_packed + (size_t)j * (size_t)klen_padded;
                                double *C_tile = C + (size_t)(bm + i) * (size_t)ldc + (size_t)(bn + j);

                                /* check offsets in bytes */
                                size_t offset_bytes = (const char*)B_tile - (const char*)B_packed;
                                size_t needed_bytes = (size_t)klen_padded * (size_t)N_tile * sizeof(double);
                                if (offset_bytes + needed_bytes > per_buf_bytes) {
                                    fprintf(stderr, "FATAL OOB PREVENT: tid=%d bm=%d bk=%d bn=%d i=%d j=%d offset_bytes=%zu needed_bytes=%zu per_buf_bytes=%zu\n",
                                            tid, bm, bk, bn, i, j, offset_bytes, needed_bytes, per_buf_bytes);
                                    fflush(stderr);
                                    /* skip dangerous call */
                                    continue;
                                }

                                micro_kernel_avx2_opt(
                                    A_tile, B_tile, C_tile,
                                    lda, klen, ldc,
                                    M_tile, N_tile, klen);
                            }
                        }
                    }
                }
            }
        }
    }

    /* free buffers */
    if (single_buffer) aligned_free_local(single_buffer);
    if (B_packed_arr) {
        for (int t=0; t<nthreads; ++t) if (B_packed_arr[t]) aligned_free_local(B_packed_arr[t]);
        free(B_packed_arr);
    }
}

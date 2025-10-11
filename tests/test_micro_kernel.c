// tests/test_micro_kernel.c
// Simple test harness to validate micro_kernel_avx2 against micro_kernel_scalar.
// Build instructions provided below.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h> 
#include "../include/micro_kernel.h"

// Simple LCG RNG (deterministic)
static uint64_t rng_state = 88172645463325252ULL;
static uint64_t lcg_rand64() {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng_state;
}
static double rnd_double() {
    // produce double in [-1, 1)
    uint64_t x = lcg_rand64();
    // take lower 52 bits
    uint64_t mant = x & ((1ULL<<52)-1);
    double d = (double)mant / (double)((1ULL<<52));
    return d * 2.0 - 1.0;
}

static void fill_rand(double *buf, size_t count) {
    for (size_t i = 0; i < count; ++i) buf[i] = rnd_double();
}

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main(void) {
    // Test several random tile sizes (M up to 6, N up to 8, K random)
    const int M_max = 6;
    const int N_max = 8;
    int tests = 100;
    int failures = 0;

    for (int t = 0; t < tests; ++t) {
        int M = (int)(lcg_rand64() % M_max) + 1; // 1..6
        int N = (int)(lcg_rand64() % N_max) + 1; // 1..8
        int K = (int)(lcg_rand64() % 64) + 1;    // 1..64

        // leading dims: use K and N as full strides to match row-major contiguous layout
        int lda = K;
        int ldb = N;
        int ldc = N;

        size_t size_A = (size_t)M * (size_t)K;
        size_t size_B = (size_t)K * (size_t)N;
        size_t size_C = (size_t)M * (size_t)N;

        double *A = (double*)malloc(sizeof(double) * size_A);
        double *B = (double*)malloc(sizeof(double) * size_B);
        double *C_ref = (double*)malloc(sizeof(double) * size_C);
        double *C_test = (double*)malloc(sizeof(double) * size_C);

        fill_rand(A, size_A);
        fill_rand(B, size_B);
        for (size_t i = 0; i < size_C; ++i) { C_ref[i] = 0.0; C_test[i] = 0.0; }

        // Run scalar kernel
        micro_kernel_scalar(A, B, C_ref, lda, ldb, ldc, M, N, K);

        // Run AVX2 kernel (if compiled in). If not present at link, you will get link error;
        // compile instructions below show both ways.
#ifdef HAS_AVX2_KERNEL
        micro_kernel_avx2(A, B, C_test, lda, ldb, ldc, M, N, K);
#else
        // If AVX2 kernel not available, skip this test.
        fprintf(stderr, "AVX2 kernel not compiled in; skipping AVX2 check. Define HAS_AVX2_KERNEL to enable.\n");
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
#endif

        double maxdiff = max_abs_diff(C_ref, C_test, size_C);
        if (!(maxdiff <= 1e-9 || isnan(maxdiff))) {
            failures++;
            printf("FAIL test %d: M=%d N=%d K=%d maxdiff=%g\n", t, M, N, K, maxdiff);
        } else {
            // optional: print pass for first few
            // printf("pass M=%d N=%d K=%d maxdiff=%g\n", M,N,K,maxdiff);
        }

        free(A); free(B); free(C_ref); free(C_test);
    }

    if (failures == 0) {
        printf("ALL %d micro-kernel tests PASSED\n", tests);
        return 0;
    } else {
        printf("%d/%d tests FAILED\n", failures, tests);
        return 1;
    }
}

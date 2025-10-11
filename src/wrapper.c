// wrapper.c
// Place in your DLL project and link with the rest of the code.
// This exports gemm_native_packed_wrapper which forwards to gemm_dispatch.

#ifdef _WIN32
  #ifdef __cplusplus
    #define EXTERN_C extern "C"
  #else
    #define EXTERN_C
  #endif
  #define EXPORT __declspec(dllexport)
#else
  #ifdef __cplusplus
    #define EXTERN_C extern "C"
  #else
    #define EXTERN_C
  #endif
  #define EXPORT
#endif

#include <stddef.h>

/* Forward-declare the exported dispatcher already present in the DLL.
   We saw 'gemm_dispatch' in the exports; treat its low-level signature as:
     void gemm_dispatch(const void* A, const void* B, void* C,
                        size_t M, size_t N, size_t K,
                        size_t lda, size_t ldb, size_t ldc);
   If your real internal signature differs, adjust size_t/int types accordingly.
*/
EXTERN_C void gemm_dispatch(
    const void* A, const void* B, void* C,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc
);

EXTERN_C EXPORT void gemm_native_packed_wrapper(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    /* Forward using the same numeric values, converting ints to size_t */
    gemm_dispatch(
        (const void*)A,
        (const void*)B,
        (void*)C,
        (size_t)M, (size_t)N, (size_t)K,
        (size_t)lda, (size_t)ldb, (size_t)ldc
    );
}

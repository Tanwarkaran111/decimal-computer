// wrapper_loader.c
// Runtime-forwarder: loads an existing gemm DLL and forwards gemm_native_packed_wrapper calls
// to its gemm_dispatch symbol.
//
// Build: gcc -shared -O3 -fPIC -I src src/wrapper_loader.c -o build/gemm_native_packed_new.dll -Wl,--export-all-symbols

#include <windows.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Typedef matching the internal dispatch signature we observed (adjust if needed)
typedef void (*gemm_dispatch_t)(
    const void* A, const void* B, void* C,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc
);

// Name of the existing DLL to load at runtime (change if you want another)
static const char *TARGET_DLL = "D:\\Invented_library\\build\\gemm_native_packed_stable320.dll";

// Cached handle & function pointer
static HMODULE cached_h = NULL;
static gemm_dispatch_t cached_fn = NULL;

// Load the target DLL and resolve the symbol (thread-safe-ish)
static int ensure_loaded(void) {
    if (cached_fn) return 1;
    if (!cached_h) {
        cached_h = LoadLibraryA(TARGET_DLL);
        if (!cached_h) return 0;
    }
    // Try common export names we found earlier
    cached_fn = (gemm_dispatch_t)GetProcAddress(cached_h, "gemm_dispatch");
    if (!cached_fn) {
        // try an alternative symbol if present
        cached_fn = (gemm_dispatch_t)GetProcAddress(cached_h, "gemm_dispatch_packed");
    }
    return cached_fn != NULL;
}

/* Exported wrapper function with a clear, stable signature for Python (float32).
   If the internal library expects double, we'll change later. */
__declspec(dllexport) void gemm_native_packed_wrapper(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    if (!ensure_loaded()) {
        // fail silently (could also set errno); returning without touching C keeps it zeroed
        return;
    }
    // Forward call (convert ints to size_t)
    cached_fn(
        (const void*)A, (const void*)B, (void*)C,
        (size_t)M, (size_t)N, (size_t)K,
        (size_t)lda, (size_t)ldb, (size_t)ldc
    );
}

#ifdef __cplusplus
}
#endif

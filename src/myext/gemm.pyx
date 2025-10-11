# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3
import cython
from cython.parallel cimport prange
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

ctypedef double DTYPE_t

# low-level block multiply kernel in Cython that runs nogil
# A and B pointers are expected to already point at the start of the block:
#   A -> &A[rowA, startK]
#   B -> &B[startK, colB]
# rowA_offset and colB_offset are absolute positions in C where results should accumulate.
cdef int _block_multiply(double* A, double* B, double* C,
                         int M, int N, int K,
                         int lda, int ldb, int ldc,
                         int rowA_offset, int colB_offset,
                         int blockM, int blockN, int blockK) nogil:
    cdef int i, j, k
    cdef double s
    cdef int a_idx, b_idx, c_idx
    # A pointer is at (rowA_offset, startK); B pointer is at (startK, colB_offset)
    for i in range(blockM):
        for j in range(blockN):
            s = 0.0
            for k in range(blockK):
                # index A relative to its pointer: A[rowA_offset + i, startK + k] -> A_ptr[i*lda + k]
                a_idx = i * lda + k
                # index B relative to its pointer: B[startK + k, colB_offset + j] -> B_ptr[k*ldb + j]
                b_idx = k * ldb + j
                s += A[a_idx] * B[b_idx]
            # write into absolute C position
            c_idx = (rowA_offset + i) * ldc + (colB_offset + j)
            C[c_idx] += s
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def gemm_openmp(np.ndarray[DTYPE_t, ndim=2] A,
                np.ndarray[DTYPE_t, ndim=2] B,
                np.ndarray[DTYPE_t, ndim=2] C,
                int blockM=64, int blockN=64, int blockK=64):
    """
    C-level batch executor: build a task list of (i_block, j_block, k_block)
    then parallelize across the flattened task list with prange. This reduces
    Python/C overhead and gives OpenMP a large pool of tasks to balance.
    """
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    cdef int lda = K
    cdef int ldb = N
    cdef int ldc = N
    cdef DTYPE_t* a = <DTYPE_t*>A.data
    cdef DTYPE_t* b = <DTYPE_t*>B.data
    cdef DTYPE_t* cptr = <DTYPE_t*>C.data

    cdef int nblocks_i = (M + blockM - 1) // blockM
    cdef int nblocks_j = (N + blockN - 1) // blockN
    cdef int nblocks_k = (K + blockK - 1) // blockK

    # total tasks = i * j * k (flattened)
    cdef Py_ssize_t total_tasks = <Py_ssize_t>nblocks_i * nblocks_j * nblocks_k
    if total_tasks == 0:
        return

    cdef Py_ssize_t t
    cdef int ib, jb, kb
    cdef int rowA, colB, startK
    cdef int curM, curN, curK

    # Parallel: iterate over task index, decode (i,j,k), run block multiply
    for t in prange(total_tasks, schedule='static', nogil=True):
        # decode t -> (ib, jb, kb)
        ib = <int>(t // (nblocks_j * nblocks_k))
        jb = <int>((t // nblocks_k) % nblocks_j)
        kb = <int>(t % nblocks_k)

        rowA = ib * blockM
        colB = jb * blockN
        startK = kb * blockK

        if rowA + blockM <= M:
            curM = blockM
        else:
            curM = M - rowA
        if colB + blockN <= N:
            curN = blockN
        else:
            curN = N - colB
        if startK + blockK <= K:
            curK = blockK
        else:
            curK = K - startK

        # Call the nogil kernel using pointers offset to block starts
        _block_multiply(
            a + rowA * lda + startK,
            b + startK * ldb + colB,
            cptr,
            M, N, K,
            lda, ldb, ldc,
            rowA, colB,
            curM, curN, curK
        )
    # end for

# python-callable block executor that releases the GIL and calls the nogil kernel
cpdef gemm_block(np.ndarray[DTYPE_t, ndim=2] A,
                 np.ndarray[DTYPE_t, ndim=2] B,
                 np.ndarray[DTYPE_t, ndim=2] C,
                 int rowA, int colB, int startK,
                 int blockM, int blockN, int blockK):
    """
    Call the nogil _block_multiply on a single tile.
    A pointer passed will be offset to &A[rowA, startK], B pointer to &B[startK, colB].
    This function releases the GIL while performing the computation.
    """
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    cdef int lda = K
    cdef int ldb = N
    cdef int ldc = N
    cdef DTYPE_t* a = <DTYPE_t*>A.data
    cdef DTYPE_t* b = <DTYPE_t*>B.data
    cdef DTYPE_t* cptr = <DTYPE_t*>C.data

    # compute current tile sizes (clamp if partial edge)
    cdef int curM = blockM if rowA + blockM <= M else M - rowA
    cdef int curN = blockN if colB + blockN <= N else N - colB
    cdef int curK = blockK if startK + blockK <= K else K - startK

    # call nogil kernel with pointers offset to the block start
    with nogil:
        _block_multiply(
            a + rowA * lda + startK,
            b + startK * ldb + colB,
            cptr,
            M, N, K,
            lda, ldb, ldc,
            rowA, colB,
            curM, curN, curK
        )
    return None

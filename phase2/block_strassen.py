# phase2/block_strassen.py
# Pure-Python Strassen (no numpy). Designed as a drop-in replacement for the
# previous block_strassen implementation but avoids numpy.array/.tolist() overhead.
#
# Notes:
# - Works with square matrices whose size is a power of two. If not, pad externally.
# - Uses a base-case cutoff (BASE_CUTOFF) where it uses an optimized schoolbook.
# - Should give better profiler numbers (no numpy conversions).

from multiprocessing import Pool

BASE_CUTOFF = 64  # tuneable: for small n, schoolbook is faster in pure Python

# -------------------------
# Basic helpers (single consistent set)
# -------------------------
def _new_matrix(n, val=0):
    return [[val] * n for _ in range(n)]


def _add(A, B):
    n = len(A)
    C = _new_matrix(n)
    for i in range(n):
        Ai, Bi, Ci = A[i], B[i], C[i]
        for j in range(n):
            Ci[j] = Ai[j] + Bi[j]
    return C


def _sub(A, B):
    n = len(A)
    C = _new_matrix(n)
    for i in range(n):
        Ai, Bi, Ci = A[i], B[i], C[i]
        for j in range(n):
            Ci[j] = Ai[j] - Bi[j]
    return C


def _split(A):
    """Split matrix A into quarters A11,A12,A21,A22. Assumes even n."""
    n = len(A)
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    return A11, A12, A21, A22


def _join(C11, C12, C21, C22):
    """Join four submatrices into one matrix."""
    top = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bottom = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bottom


def _pad_to_power_of_two(A, size=None):
    """Pad square matrix A to next power-of-two (or to 'size' if given)."""
    n = len(A)
    if size is None:
        # next power of two >= n
        m = 1 << ((n - 1).bit_length()) if n & (n - 1) else n
    else:
        m = size
    if m == n:
        return [row[:] for row in A]
    P = _new_matrix(m)
    for i in range(n):
        P[i][:n] = A[i][:]
    return P


def _unpad(C, n):
    """Return top-left n x n block of C"""
    return [row[:n] for row in C[:n]]


# -------------------------
# Schoolbook variants
# -------------------------
def _schoolbook(A, B):
    """Plain schoolbook (fallback). Accepts matrix or scalar inputs."""
    # Scalar shortcut: if both args are scalars (no __len__), return numeric product
    if not hasattr(A, "__len__") and not hasattr(B, "__len__"):
        return A * B

    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C



def _schoolbook_blocked(A, B, block_size: int = 16):
    """
    Blocked (tiled) schoolbook matrix multiplication.
    Pure Python, cache-friendly version.

    Accepts either matrix inputs or scalar ints (scalar -> numeric product).
    """
    # Scalar shortcut: if both args are scalars (no __len__), return numeric product
    if not hasattr(A, "__len__") and not hasattr(B, "__len__"):
        return A * B

    n = len(A)
    C = [[0] * n for _ in range(n)]
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    Ai = A[i]
                    Ci = C[i]
                    for k in range(kk, min(kk + block_size, n)):
                        aik = Ai[k]
                        Bk = B[k]
                        for j in range(jj, min(jj + block_size, n)):
                            Ci[j] += aik * Bk[j]
    return C



# -------------------------
# Strassen recursive core
# -------------------------
def _strassen_recursive(A, B, cutoff=BASE_CUTOFF):
    """Recursive Strassen which assumes A,B are square and length is power of two."""
    n = len(A)
    if n <= cutoff:
        return _schoolbook(A, B)

    # split into quarters
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)

    # compute the 7 products (recursively), always passing cutoff
    M1 = _strassen_recursive(_add(A11, A22), _add(B11, B22), cutoff)
    M2 = _strassen_recursive(_add(A21, A22), B11, cutoff)
    M3 = _strassen_recursive(A11, _sub(B12, B22), cutoff)
    M4 = _strassen_recursive(A22, _sub(B21, B11), cutoff)
    M5 = _strassen_recursive(_add(A11, A12), B22, cutoff)
    M6 = _strassen_recursive(_sub(A21, A11), _add(B11, B12), cutoff)
    M7 = _strassen_recursive(_sub(A12, A22), _add(B21, B22), cutoff)

    # assemble result quarters
    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    return _join(C11, C12, C21, C22)


# -------------------------
# Public wrapper (scalar shortcut + padding)
# -------------------------
def block_strassen(A, B, base_cutoff=BASE_CUTOFF):
    """
    Public wrapper: pads inputs to power-of-two, calls recursive Strassen with cutoff,
    and unpads the result. Accepts either matrix inputs (list-of-lists) or scalar
    integer inputs (compatibility shim).
    """
    # --- scalar shortcut ---
    # If called with scalar integer arguments (the integration test calls
    # algorithms with ints), handle that quickly and return numeric product.
    if not hasattr(A, "__len__") and not hasattr(B, "__len__"):
        return A * B
    # ------------------------

    # sanity shapes (assume square nxn for gemm scenarios here)
    n = len(A)
    if n == 0:
        return []

    # pad to power-of-two size >= n
    m = 1 << ((n - 1).bit_length()) if n & (n - 1) else n
    if m != n:
        Ap = _pad_to_power_of_two(A, m)
        Bp = _pad_to_power_of_two(B, m)
    else:
        Ap = [row[:] for row in A]
        Bp = [row[:] for row in B]

    # call recursive strassen with cutoff argument
    Cpad = _strassen_recursive(Ap, Bp, cutoff=base_cutoff)

    # unpad to original size
    Cres = _unpad(Cpad, n)
    return Cres


# -------------------------
# Parallel variant (optional)
# -------------------------
def _strassen_worker(args):
    # args: (A_sub, B_sub, cutoff)
    A_sub, B_sub, cutoff = args
    return _strassen_recursive(A_sub, B_sub, cutoff)


def block_strassen_parallel(A, B, base_cutoff=64, procs=4):
    """
    Parallel top-level Strassen: splits inputs, spawns up to `procs` workers
    to compute the 7 M-products in parallel, then assembles result.
    Uses same padding/unpadding as block_strassen.
    """
    n = len(A)
    if n == 0:
        return []

    # pad to power-of-two
    m = 1 << ((n - 1).bit_length()) if n & (n - 1) else n
    if m != n:
        Ap = _pad_to_power_of_two(A, m)
        Bp = _pad_to_power_of_two(B, m)
    else:
        Ap = [row[:] for row in A]
        Bp = [row[:] for row in B]

    # if already small, use the same recursive implementation directly
    if m <= base_cutoff or procs <= 1:
        Cpad = _strassen_recursive(Ap, Bp, cutoff=base_cutoff)
        return _unpad(Cpad, n)

    # compute quarters
    A11, A12, A21, A22 = _split(Ap)
    B11, B12, B21, B22 = _split(Bp)

    # prepare the 7 tasks (A_sub, B_sub, cutoff)
    tasks = [
        (_add(A11, A22), _add(B11, B22), base_cutoff),  # M1
        (_add(A21, A22), B11, base_cutoff),             # M2
        (A11, _sub(B12, B22), base_cutoff),             # M3
        (A22, _sub(B21, B11), base_cutoff),             # M4
        (_add(A11, A12), B22, base_cutoff),             # M5
        (_sub(A21, A11), _add(B11, B12), base_cutoff),  # M6
        (_sub(A12, A22), _add(B21, B22), base_cutoff),  # M7
    ]

    with Pool(processes=min(procs, len(tasks))) as pool:
        results = pool.map(_strassen_worker, tasks)

    M1, M2, M3, M4, M5, M6, M7 = results

    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    Cpad = _join(C11, C12, C21, C22)
    return _unpad(Cpad, n)


# -------------------------
# Compatibility shim for tests expecting run_strassen
# -------------------------
try:
    run_strassen  # type: ignore
except NameError:
    if "block_strassen" in globals() and callable(globals()["block_strassen"]):
        def run_strassen(A, B, *args, **kwargs):
            return block_strassen(A, B, *args, **kwargs)
    else:
        def run_strassen(*args, **kwargs):
            raise NotImplementedError("run_strassen compatibility shim: implement or export run_strassen")

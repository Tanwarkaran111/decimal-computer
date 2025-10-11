# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
parallel_cworker.pyx - Hybrid backend (int64 fast path + big-int delegated to decimal_computer.auto_bigint)

Fast int64 path when values fit; fallback delegates to the Python auto_bigint connector for big-int ops.
"""

from cpython.array cimport array as carray
import array
cimport cython

# Import Python-level auto_bigint connector (router to gmpy2/fft/karatsuba)
# We import lazily inside functions to avoid import-time costs during cythonize in some setups,
# but importing here is also fine when auto_bigint.py exists.
import importlib

# Optional NumPy
try:
    import numpy as np
except Exception:
    np = None

# typedef
ctypedef long long int64_t

# -------------------------
# Helper: try to convert to array('q') without overflow
# -------------------------
@cython.inline
cdef carray _try_array_q_from_iterable(object obj) except *:
    """
    Try to convert obj into array.array('q') (signed 64-bit) without overflow.
    Raises OverflowError if any value doesn't fit in int64.
    """
    cdef object a_out_py
    cdef object cobj
    cdef object casted
    cdef Py_ssize_t n, i
    cdef object it
    cdef object val

    # Fast path: already array.array
    if isinstance(obj, array.array):
        if obj.typecode == 'q':
            return <carray>obj
        a_out_py = array.array('q')
        a_out_py.extend(obj)  # may raise OverflowError
        return <carray>a_out_py

    # NumPy fast-path
    if np is not None:
        try:
            if isinstance(obj, np.ndarray):
                # int64 dtype -> bulk copy
                if obj.dtype == np.int64:
                    cobj = obj if obj.flags.c_contiguous else obj.ravel(order='C')
                    a_out_py = array.array('q')
                    a_out_py.frombytes(cobj.tobytes(order='C'))
                    return <carray>a_out_py
                # other integer dtype -> cast to int64 (may copy)
                if np.issubdtype(obj.dtype, np.integer):
                    casted = obj.astype(np.int64, copy=False)
                    cobj = casted if casted.flags.c_contiguous else casted.ravel(order='C')
                    a_out_py = array.array('q')
                    a_out_py.frombytes(cobj.tobytes(order='C'))
                    return <carray>a_out_py
        except OverflowError:
            raise
        except Exception:
            # fall through to generic handling
            pass

    # Generic iterable: use sequence-index path if possible
    a_out_py = array.array('q')

    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        n = len(obj)
        for i in range(n):
            val = obj[i]
            a_out_py.append(<int64_t>val)  # may raise OverflowError
        return <carray>a_out_py

    # Generic iterator fallback
    it = iter(obj)
    while True:
        try:
            val = next(it)
        except StopIteration:
            break
        a_out_py.append(<int64_t>val)  # may raise OverflowError

    return <carray>a_out_py

# -------------------------
# Helper: convert to Python list of ints (big-int)
# (kept for internal use but main big-int work will delegate to auto_bigint)
# -------------------------
@cython.inline
cdef list _to_pylist_of_ints(object obj):
    cdef list out
    cdef object mv
    cdef Py_ssize_t i

    if isinstance(obj, list):
        return [int(x) for x in obj]

    if isinstance(obj, array.array):
        return [int(x) for x in obj]

    if np is not None:
        try:
            if isinstance(obj, np.ndarray):
                out = obj.tolist()
                if isinstance(out, list):
                    return [int(x) for x in out]
                else:
                    return [int(out)]
        except Exception:
            pass

    try:
        mv = memoryview(obj)
    except Exception:
        mv = None

    if mv is not None:
        if hasattr(mv, "ndim") and mv.ndim == 1:
            return [int(x) for x in mv]
        elif hasattr(mv, "ndim") and mv.ndim == 0:
            try:
                return [int(mv.item())]
            except Exception:
                return [int(mv[()])]
        else:
            out = []
            for row in mv:
                out.extend([int(x) for x in row])
            return out

    if hasattr(obj, "__iter__"):
        return [int(x) for x in obj]

    try:
        return [int(obj)]
    except Exception:
        raise TypeError("Cannot convert object to list of ints")

# -------------------------
# int64 C-speed implementations (internal helpers)
# -------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _mul_int64_full_c(carray a_arr, carray b_arr):
    cdef Py_ssize_t n_a = len(a_arr)
    cdef Py_ssize_t n_b = len(b_arr)
    cdef Py_ssize_t out_n
    cdef object out_py
    cdef carray out
    cdef int64_t[::1] va
    cdef int64_t[::1] vb
    cdef int64_t[::1] vout
    cdef Py_ssize_t i

    if n_a != n_b:
        raise ValueError("input buffers must have same length")

    out_n = n_a
    out_py = array.array('q', [0]) * out_n
    out = <carray>out_py

    va = a_arr
    vb = b_arr
    vout = out

    for i in range(out_n):
        vout[i] = va[i] * vb[i]

    return out_py

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _mul_int64_slice_c(carray a_arr, carray b_arr, Py_ssize_t start, Py_ssize_t end):
    cdef Py_ssize_t n_a = len(a_arr)
    cdef Py_ssize_t n_b = len(b_arr)
    cdef Py_ssize_t out_n
    cdef object out_py
    cdef carray out
    cdef int64_t[::1] va
    cdef int64_t[::1] vb
    cdef int64_t[::1] vout
    cdef Py_ssize_t i

    if start < 0 or end < start:
        raise IndexError("invalid slice bounds")
    if end > n_a or end > n_b:
        raise IndexError("slice end out of range for input buffers")

    out_n = end - start
    out_py = array.array('q', [0]) * out_n
    out = <carray>out_py

    va = a_arr
    vb = b_arr
    vout = out

    for i in range(out_n):
        vout[i] = va[start + i] * vb[start + i]

    return out_py

# -------------------------
# big-int Python implementations (delegate to auto_bigint)
# -------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object mul_big_full(object a_obj, object b_obj):
    """
    Delegate big-int full-vector multiply to decimal_computer.auto_bigint.mul_bigint_list.
    Returns a Python list of ints (auto_bigint handles strategy selection).
    """
    # Import the connector module (import time small; module already in project)
    ab = importlib.import_module("decimal_computer.auto_bigint")
    # Convert inputs to Python lists of ints (auto_bigint will re-convert as needed)
    a_list = _to_pylist_of_ints(a_obj)
    b_list = _to_pylist_of_ints(b_obj)
    # Delegate to auto_bigint (returns list of Python ints)
    return ab.mul_bigint_list(a_list, b_list, strategy="auto")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object mul_big_slice(object a_obj, object b_obj, Py_ssize_t start, Py_ssize_t end):
    """
    Delegate big-int slice multiply to decimal_computer.auto_bigint.mul_bigint_slice.
    """
    ab = importlib.import_module("decimal_computer.auto_bigint")
    # Let auto_bigint handle slicing/conversion — it expects sequences/slices
    return ab.mul_bigint_slice(a_obj, b_obj, start, end, strategy="auto")

# -------------------------
# Public hybrid API (auto-selecting)
# -------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object mul_int64_full(object a_obj, object b_obj):
    """
    Hybrid multiply full arrays: try int64 path first; if overflow or incompatible types,
    fall back to big-int path (now delegated to auto_bigint).
    Returns array.array('q') if int64 path used, else list of Python ints.
    """
    try:
        a_arr = _try_array_q_from_iterable(a_obj)
        b_arr = _try_array_q_from_iterable(b_obj)
        return _mul_int64_full_c(a_arr, b_arr)
    except OverflowError:
        return mul_big_full(a_obj, b_obj)
    except Exception:
        return mul_big_full(a_obj, b_obj)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object mul_int64_slice(object a_obj, object b_obj, Py_ssize_t start, Py_ssize_t end):
    """
    Hybrid slice multiply. Uses C int64 path when possible; otherwise delegates to auto_bigint.
    """
    try:
        a_arr = _try_array_q_from_iterable(a_obj)
        b_arr = _try_array_q_from_iterable(b_obj)
        return _mul_int64_slice_c(a_arr, b_arr, start, end)
    except OverflowError:
        return mul_big_slice(a_obj, b_obj, start, end)
    except Exception:
        return mul_big_slice(a_obj, b_obj, start, end)

# alias
mul_int64_range = mul_int64_slice

# -------------------------
# Utilities
# -------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object to_array_q_if_possible(object obj):
    return _try_array_q_from_iterable(obj)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Py_ssize_t len_maybe_array_q(object obj):
    try:
        a = _try_array_q_from_iterable(obj)
        return len(a)
    except Exception:
        l = _to_pylist_of_ints(obj)
        return len(l)

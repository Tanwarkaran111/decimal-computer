# distutils: language = c
# fast_math.pyx - Cython digit arithmetic for Phase2

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset

@cython.boundscheck(False)
@cython.wraparound(False)
def digit_mul(list a, list b):
    """
    Multiply two integers represented as lists of decimal digits (MSB-first).
    Returns a Python list of digits (MSB-first).
    Example: [1,2,3] * [4,5] = [5,5,3,5] (i.e. 5535).
    """
    cdef Py_ssize_t na = len(a)
    cdef Py_ssize_t nb = len(b)
    cdef Py_ssize_t nc = na + nb
    cdef int *res = <int *> malloc(nc * cython.sizeof(int))
    if not res:
        raise MemoryError()
    memset(res, 0, nc * cython.sizeof(int))

    cdef int carry, mul
    cdef Py_ssize_t i, j, k

    for i in range(na - 1, -1, -1):
        for j in range(nb - 1, -1, -1):
            k = i + j + 1
            mul = (<int>a[i]) * (<int>b[j]) + res[k]
            res[k] = mul % 10
            res[k - 1] += mul // 10

    # Convert back to Python list
    out = [res[i] for i in range(nc)]
    free(res)

    # Remove leading zeros
    while len(out) > 1 and out[0] == 0:
        out.pop(0)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def digit_add(list a, list b):
    """
    Add two integers represented as digit lists (MSB-first).
    """
    # pad shorter one
    if len(a) < len(b):
        a = [0] * (len(b) - len(a)) + a
    elif len(b) < len(a):
        b = [0] * (len(a) - len(b)) + b

    cdef Py_ssize_t n = len(a)
    cdef int carry = 0
    cdef int s
    out = [0] * (n + 1)

    for i in range(n - 1, -1, -1):
        s = (<int>a[i]) + (<int>b[i]) + carry
        out[i + 1] = s % 10
        carry = s // 10

    out[0] = carry
    while len(out) > 1 and out[0] == 0:
        out.pop(0)
    return out

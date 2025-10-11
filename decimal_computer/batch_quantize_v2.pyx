# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from cpython.array cimport array as carray
from array import array
cimport cython
from libc.stdint cimport int64_t
from libc.stdlib cimport llabs
from libc.math cimport llround

cdef inline int64_t pow10_int(int d) nogil:
    cdef int64_t p = 1
    cdef int i
    for i in range(d):
        p *= 10
    return p

cdef inline int64_t round_div_int64(int64_t num, int64_t den, int rounding_mode) nogil:
    # rounding_mode: 0 -> HALF_UP, 1 -> HALF_EVEN
    # num and den positive expected here (we handle sign outside)
    if den == 0:
        return 0
    cdef int64_t q = num // den
    cdef int64_t r = num - q * den
    if r == 0:
        return q
    # compare 2*r vs den to decide half
    cdef int cmp = 2 * r - den
    if cmp > 0:
        return q + 1
    elif cmp < 0:
        return q
    else:
        # exact half
        if rounding_mode == 0:
            return q + 1
        else:
            # HALF_EVEN: if q is odd, round up
            return q + (q & 1)

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_quantize_v2(a_ints, b_ints, a_sc, b_sc, int target_scale, rounding: str):
    """
    Fast path: accepts Python sequences that support buffer protocol (array.array of 'q' or lists of ints).
    Tries to perform int64_t path (nogil) when all intermediate values fit in int64 range
    and scale differences are reasonable. Returns array('q').
    Falls back to raising ValueError when unable to process (caller can fall back to Python safe path).
    """
    cdef Py_ssize_t n = len(a_ints)
    if n != len(b_ints) or n != len(a_sc) or n != len(b_sc):
        raise ValueError("input lengths must match")

    # Convert rounding to mode
    cdef int rounding_mode = 0
    if rounding == "ROUND_HALF_EVEN":
        rounding_mode = 1
    else:
        rounding_mode = 0

    out = array('q')

    # try to get raw buffer views if inputs are array('q') or similar
    # We'll fallback to Python-level loop if any value is out-of-range for int64 scaled path.
    cdef Py_ssize_t i
    cdef int64_t ai, bi
    cdef int asc, bsc
    cdef long long prod
    for i in range(n):
        try:
            ai = <int64_t>a_ints[i]
            bi = <int64_t>b_ints[i]
            asc = int(a_sc[i])
            bsc = int(b_sc[i])
        except Exception:
            raise ValueError("Inputs must be integer sequences convertible to int64 for fast path")

        # orig scale
        cdef int orig_scale = asc + bsc
        cdef int d = orig_scale - target_scale

        # compute product in 128-bit? Cython doesn't have builtin, so detect overflow possibility:
        # We conservatively check magnitude to avoid overflow: if |ai| or |bi| too large, bail.
        # Here we assume int64_t multiply fits if abs(ai) <= 2^31 and abs(bi) <= 2^31 etc.
        # Use a conservative threshold: 1<<31 (~2.1e9) -> product fits in 63 bits.
        if llabs(ai) > (1 << 31) or llabs(bi) > (1 << 31):
            raise ValueError("value too large for fast int64 path")

        prod = ai * bi  # fits if above checks pass

        if d == 0:
            out.append(<int64_t>prod)
            continue
        elif d > 0:
            # need to divide by 10**d with rounding
            if d > 18:
                raise ValueError("scale difference too large for fast path")
            cdef int64_t shift = pow10_int(d)
            cdef int sign = 1
            cdef unsigned long long unum
            if prod < 0:
                sign = -1
                unum = <unsigned long long>(-prod)
            else:
                unum = <unsigned long long>prod
            cdef int64_t q = round_div_int64(<int64_t>unum, shift, rounding_mode)
            if sign < 0:
                q = -q
            out.append(q)
        else:
            # d < 0: need to multiply by 10**(-d)
            dd = -d
            if dd > 9:
                # multiplying by large power risks overflow
                raise ValueError("scale increase too large for fast path")
            cdef int64_t mul = pow10_int(dd)
            cdef long long val = prod * mul
            out.append(<int64_t>val)

    return out

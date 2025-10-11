# decimal_computer/_c_quantize.pyx
# cython: language_level=3
from cpython.array cimport array as pyarray
cimport cython

# rounding codes
ROUND_HALF_EVEN = 0
ROUND_HALF_UP = 1

# 64-bit limits
cdef long long MAX_INT64 = 9223372036854775807
cdef long long MIN_INT64 = -9223372036854775808

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _c_mul_and_quantize_from_arrays_impl(
    object a_array,
    object b_array,
    int orig_scale,
    int target_scale,
    int rounding_code
):
    cdef long long[:] a_mv = a_array  # type: ignore
    cdef long long[:] b_mv = b_array  # type: ignore
    cdef Py_ssize_t n = a_mv.shape[0]
    if n != b_mv.shape[0]:
        raise ValueError("a and b arrays must have same length")

    cdef pyarray res = pyarray("q")
    res.extend([0] * n)
    cdef long long[:] res_mv = res  # type: ignore

    cdef pyarray mask = pyarray("b")
    mask.extend([0] * n)
    cdef signed char[:] mask_mv = mask  # type: ignore

    cdef int delta = orig_scale - target_scale
    cdef long long factor10 = 1
    cdef long long mul_factor = 1
    cdef long long half = 0

    cdef Py_ssize_t i, j
    cdef long long a_val, b_val, prod, q, r, abs_r, out_int

    # prepare scaling factors
    if delta > 0:
        factor10 = 1
        for j in range(delta):
            if factor10 > MAX_INT64 // 10:
                for i in range(n):
                    mask_mv[i] = 1
                return res, mask
            factor10 *= 10
        half = factor10 // 2
    elif delta < 0:
        mul_factor = 1
        for j in range(-delta):
            if mul_factor > MAX_INT64 // 10:
                for i in range(n):
                    mask_mv[i] = 1
                return res, mask
            mul_factor *= 10

    for i in range(n):
        a_val = a_mv[i]
        b_val = b_mv[i]

        # multiplication overflow pre-check
        if a_val != 0 and b_val != 0:
            if b_val > 0:
                if a_val > MAX_INT64 // b_val or a_val < MIN_INT64 // b_val:
                    mask_mv[i] = 1
                    continue
            else:
                if a_val < MAX_INT64 // b_val or a_val > MIN_INT64 // b_val:
                    mask_mv[i] = 1
                    continue

        prod = <long long>(a_val * b_val)

        if delta == 0:
            out_int = prod
        elif delta < 0:
            if prod != 0 and (prod > MAX_INT64 // mul_factor or prod < MIN_INT64 // mul_factor):
                mask_mv[i] = 1
                continue
            out_int = prod * mul_factor
        else:
            # use Python-style floor division for q and r so rounding matches Python
            if prod >= 0:
                q = prod // factor10
                r = prod - q * factor10
            else:
                q = - ((-prod) // factor10)
                r = prod - q * factor10

            if rounding_code == ROUND_HALF_UP:
                if prod >= 0:
                    if r >= half:
                        q += 1
                else:
                    if r <= -half:
                        q -= 1
                out_int = q
            else:  # ROUND_HALF_EVEN
                abs_r = r if r >= 0 else -r
                if abs_r * 2 == factor10:
                    if (q & 1) != 0:
                        if prod >= 0:
                            q += 1
                        else:
                            q -= 1
                    out_int = q
                else:
                    if prod >= 0:
                        if r * 2 >= factor10:
                            q += 1
                    else:
                        if (-r) * 2 >= factor10:
                            q -= 1
                    out_int = q

        res_mv[i] = out_int

    return res, mask


@cython.boundscheck(False)
@cython.wraparound(False)
def c_mul_and_quantize_from_arrays(*args):
    if len(args) == 5:
        a_array, b_array, orig_scale, target_scale, rounding_code = args
        return _c_mul_and_quantize_from_arrays_impl(a_array, b_array, int(orig_scale), int(target_scale), int(rounding_code))
    elif len(args) == 6:
        a_array, b_array, a_scale, b_scale, target_scale, rounding_code = args
        orig_scale = int(a_scale) + int(b_scale)
        return _c_mul_and_quantize_from_arrays_impl(a_array, b_array, orig_scale, int(target_scale), int(rounding_code))
    else:
        raise TypeError("c_mul_and_quantize_from_arrays expects 5 or 6 arguments (got %d)" % len(args))


# add kernel (same floor-division fix applied)
@cython.boundscheck(False)
@cython.wraparound(False)
def c_add_and_quantize_from_arrays(
    object a_array, object b_array, int src_scale, int target_scale, int rounding_code
):
    cdef long long[:] a_mv = a_array  # type: ignore
    cdef long long[:] b_mv = b_array  # type: ignore
    cdef Py_ssize_t n = a_mv.shape[0]
    if n != b_mv.shape[0]:
        raise ValueError("a and b arrays must have same length")

    cdef pyarray res = pyarray("q")
    res.extend([0] * n)
    cdef long long[:] res_mv = res  # type: ignore

    cdef pyarray mask = pyarray("b")
    mask.extend([0] * n)
    cdef signed char[:] mask_mv = mask  # type: ignore

    cdef int delta = src_scale - target_scale
    cdef long long factor10 = 1
    cdef long long mul_factor = 1
    cdef long long half = 0

    cdef Py_ssize_t i, j
    cdef long long a_val, b_val, s, q, r, abs_r, out_int

    if delta > 0:
        factor10 = 1
        for j in range(delta):
            if factor10 > MAX_INT64 // 10:
                for i in range(n):
                    mask_mv[i] = 1
                return res, mask
            factor10 *= 10
        half = factor10 // 2
    elif delta < 0:
        mul_factor = 1
        for j in range(-delta):
            if mul_factor > MAX_INT64 // 10:
                for i in range(n):
                    mask_mv[i] = 1
                return res, mask
            mul_factor *= 10

    for i in range(n):
        a_val = a_mv[i]
        b_val = b_mv[i]
        s = a_val + b_val

        if (b_val > 0 and s < a_val) or (b_val < 0 and s > a_val):
            mask_mv[i] = 1
            continue

        if delta == 0:
            out_int = s
        elif delta < 0:
            if s != 0 and (s > MAX_INT64 // mul_factor or s < MIN_INT64 // mul_factor):
                mask_mv[i] = 1
                continue
            out_int = s * mul_factor
        else:
            # floor-division style
            if s >= 0:
                q = s // factor10
                r = s - q * factor10
            else:
                q = - ((-s) // factor10)
                r = s - q * factor10

            if rounding_code == ROUND_HALF_UP:
                if s >= 0:
                    if r >= half:
                        q += 1
                else:
                    if r <= -half:
                        q -= 1
                out_int = q
            else:  # HALF_EVEN
                abs_r = r if r >= 0 else -r
                if abs_r * 2 == factor10:
                    if (q & 1) != 0:
                        if s >= 0:
                            q += 1
                        else:
                            q -= 1
                    out_int = q
                else:
                    if s >= 0:
                        if r * 2 >= factor10:
                            q += 1
                    else:
                        if (-r) * 2 >= factor10:
                            q -= 1
                    out_int = q
        res_mv[i] = out_int

    return res, mask


# sub kernel (same floor-division fix)
@cython.boundscheck(False)
@cython.wraparound(False)
def c_sub_and_quantize_from_arrays(
    object a_array, object b_array, int src_scale, int target_scale, int rounding_code
):
    cdef long long[:] a_mv = a_array  # type: ignore
    cdef long long[:] b_mv = b_array  # type: ignore
    cdef Py_ssize_t n = a_mv.shape[0]
    if n != b_mv.shape[0]:
        raise ValueError("a and b arrays must have same length")

    cdef pyarray res = pyarray("q")
    res.extend([0] * n)
    cdef long long[:] res_mv = res  # type: ignore

    cdef pyarray mask = pyarray("b")
    mask.extend([0] * n)
    cdef signed char[:] mask_mv = mask  # type: ignore

    cdef int delta = src_scale - target_scale
    cdef long long factor10 = 1
    cdef long long mul_factor = 1
    cdef long long half = 0

    cdef Py_ssize_t i, j
    cdef long long a_val, b_val, d, q, r, abs_r, out_int

    if delta > 0:
        factor10 = 1
        for j in range(delta):
            if factor10 > MAX_INT64 // 10:
                for i in range(n):
                    mask_mv[i] = 1
                return res, mask
            factor10 *= 10
        half = factor10 // 2
    elif delta < 0:
        mul_factor = 1
        for j in range(-delta):
            if mul_factor > MAX_INT64 // 10:
                for i in range(n):
                    mask_mv[i] = 1
                return res, mask
            mul_factor *= 10

    for i in range(n):
        a_val = a_mv[i]
        b_val = b_mv[i]
        d = a_val - b_val

        if (b_val < 0 and d < a_val) or (b_val > 0 and d > a_val):
            mask_mv[i] = 1
            continue

        if delta == 0:
            out_int = d
        elif delta < 0:
            if d != 0 and (d > MAX_INT64 // mul_factor or d < MIN_INT64 // mul_factor):
                mask_mv[i] = 1
                continue
            out_int = d * mul_factor
        else:
            # floor-division style
            if d >= 0:
                q = d // factor10
                r = d - q * factor10
            else:
                q = - ((-d) // factor10)
                r = d - q * factor10

            if rounding_code == ROUND_HALF_UP:
                if d >= 0:
                    if r >= half:
                        q += 1
                else:
                    if r <= -half:
                        q -= 1
                out_int = q
            else:
                abs_r = r if r >= 0 else -r
                if abs_r * 2 == factor10:
                    if (q & 1) != 0:
                        if d >= 0:
                            q += 1
                        else:
                            q -= 1
                    out_int = q
                else:
                    if d >= 0:
                        if r * 2 >= factor10:
                            q += 1
                    else:
                        if (-r) * 2 >= factor10:
                            q -= 1
                    out_int = q
        res_mv[i] = out_int

    return res, mask

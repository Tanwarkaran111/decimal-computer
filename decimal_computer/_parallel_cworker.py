from . import parallel_cworker as _pc
from array import array

def mul_int64_any(a_obj, b_obj):
    """
    Accepts: array('q'), numpy.ndarray(dtype=int64), or any iterable of ints
    Returns: Python list of int results (fast C path)
    """
    if isinstance(a_obj, array) and a_obj.typecode == 'q' and isinstance(b_obj, array) and b_obj.typecode == 'q':
        return _pc.mul_int64_buffers(a_obj, b_obj)

    try:
        import numpy as np
        if isinstance(a_obj, np.ndarray) and a_obj.dtype == np.int64 and isinstance(b_obj, np.ndarray) and b_obj.dtype == np.int64:
            return _pc.mul_int64_buffers(a_obj, b_obj)
    except Exception:
        pass

    A = array('q')
    B = array('q')
    for ai in a_obj:
        v = int(ai)
        if v < -(1 << 63) or v > (1 << 63) - 1:
            raise OverflowError("value too large for int64 fast path")
        A.append(v)
    for bi in b_obj:
        v = int(bi)
        if v < -(1 << 63) or v > (1 << 63) - 1:
            raise OverflowError("value too large for int64 fast path")
        B.append(v)

    return _pc.mul_int64_buffers(A, B)

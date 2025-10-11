# cython: language_level=3
"""
Simple Cython batch quantize prototype.
Accepts Python sequences (lists or array.array of ints) and returns array('q')
of quantized integers using the same quantize_int routine from decimal_computer.rounding.
This is intentionally minimal and uses Python big ints for correctness.
"""
from libc.stdlib cimport malloc, free
import array
from typing import Any

# Import quantize_int from the package (pure-Python function)
from decimal_computer.rounding import quantize_int as _quantize_int

def batch_quantize(a_ints, b_ints, a_scales, b_scales, int target_scale, rounding) -> "array.array":
    """
    Compute quantized results for product of pairs:
      for i: prod = int(a_ints[i]) * int(b_ints[i])
      orig_scale = int(a_scales[i]) + int(b_scales[i])
      q = quantize_int(prod, orig_scale=orig_scale, target_scale=target_scale, rounding=rounding)
      append int(q)

    Returns array('q') of ints.
    This routine uses Python big-int multiply and quantize_int but reduces Python call overhead
    compared to a pure-Python outer loop by keeping the loop inside compiled Cython code.
    """
    if len(a_ints) != len(b_ints) or len(a_ints) != len(a_scales) or len(a_ints) != len(b_scales):
        raise ValueError("input sequences must have same length")

    n = len(a_ints)
    out = array.array("q")
    qfunc = _quantize_int
    for i in range(n):
        ai = a_ints[i]
        bi = b_ints[i]
        prod = ai * bi
        orig_scale = int(a_scales[i]) + int(b_scales[i])
        q = qfunc(prod, orig_scale=orig_scale, target_scale=target_scale, rounding=rounding)
        out.append(int(q))
    return out

"""
Public shim for batch quantize.
Try to import the compiled Cython extension (_batch_quantize.batch_quantize).
If unavailable, provide a pure-Python fallback that mirrors the previous logic.
"""
from __future__ import annotations
import array
from typing import Sequence

try:
    from ._batch_quantize import batch_quantize as _c_batch_quantize  # type: ignore
except Exception:
    _c_batch_quantize = None

from .rounding import quantize_int

def batch_quantize(a_ints: Sequence[int],
                   b_ints: Sequence[int],
                   a_scales: Sequence[int],
                   b_scales: Sequence[int],
                   target_scale: int,
                   rounding) -> array.array:
    """
    Wrapper that uses compiled batch_quantize if available, otherwise falls back
    to a safe Python implementation using quantize_int.
    Input sequences must be same length.
    Returns array('q') of quantized ints.
    """
    if _c_batch_quantize is not None:
        return _c_batch_quantize(a_ints, b_ints, a_scales, b_scales, int(target_scale), rounding)

    if len(a_ints) != len(b_ints) or len(a_ints) != len(a_scales) or len(a_ints) != len(b_scales):
        raise ValueError("input sequences must have same length")

    out = array.array("q")
    qfunc = quantize_int
    for ai, bi, ascl, bscl in zip(a_ints, b_ints, a_scales, b_scales):
        prod = int(ai) * int(bi)
        orig_scale = int(ascl) + int(bscl)
        q = qfunc(prod, orig_scale=orig_scale, target_scale=int(target_scale), rounding=rounding)
        out.append(int(q))
    return out

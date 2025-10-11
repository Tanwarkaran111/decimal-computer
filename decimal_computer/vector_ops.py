# decimal_computer/vector_ops.py
"""
Vectorized / batched arithmetic helpers for FastDecimal with integer-array fast paths.

Micro-optimized _mul_ints/_add_ints/_sub_ints and an optimized broadcast path
for scalar multipliers to reduce Python-level overhead while preserving behavior.
"""
from __future__ import annotations
from typing import Iterable, Iterator, List, Sequence, Union, Any, Tuple
from decimal import Decimal
from array import array
import logging

from .fastdecimal import FastDecimal, FastDecimalView
from .context import get_context
from .rounding import quantize_int

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

Scalar = Union[int, float, str, FastDecimal]


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple))


def ensure_fastdec(x: Scalar) -> FastDecimal:
    """Convert common types to FastDecimal using current context."""
    if isinstance(x, FastDecimal):
        return x
    if isinstance(x, int):
        return FastDecimal.from_int(x)
    if isinstance(x, float):
        return FastDecimal.from_float(x)
    if isinstance(x, str):
        return FastDecimal.from_str(x)
    # fallback: try str conversion
    return FastDecimal.from_str(str(x))


def _pairwise_or_broadcast(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> Iterator[Tuple[FastDecimal, FastDecimal]]:
    if _is_sequence(b):
        b_list = list(b)  # allow any sequence
        if len(b_list) != len(a_list):
            raise ValueError("When both arguments are sequences, lengths must match")
        for ai, bi in zip(a_list, b_list):
            yield ensure_fastdec(ai), ensure_fastdec(bi)
    else:
        b_scalar = ensure_fastdec(b)
        for ai in a_list:
            yield ensure_fastdec(ai), b_scalar


def mul_vectors_raw(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> array:
    """
    High-performance path: returns an array('q') of quantized integer results (FastDecimal.int_value)
    for elementwise multiplication (supports scalar broadcast).
    - When both inputs are sequences of FastDecimal (pure ints) this uses the compiled C path.
    - If any step cannot use the compiled fast path (overflow, mixed orig_decimal, mismatched scales),
      falls back to Python-produced array via quantize_int.
    Notes:
      - Returned values are integers already quantized to the active context.scale.
      - To get FastDecimal objects, convert each int via FastDecimal(int_value=..., scale=ctx.scale).
    """
    ctx = get_context()

    def _ensure_fd_list(seq):
        return [ensure_fastdec(x) for x in seq]

    # broadcast scalar case
    if not _is_sequence(b):
        b_fd = ensure_fastdec(b)
        a_fd_list = _ensure_fd_list(a_list)

        # try the fast array path only when both sides are pure integer-backed and same scale
        if b_fd._orig_decimal is None and all(fd._orig_decimal is None and fd.scale == a_fd_list[0].scale for fd in a_fd_list):
            try:
                a_arr = array("q", (fd.int_value for fd in a_fd_list))
                b_arr_rep = array("q", [b_fd.int_value] * len(a_fd_list))
                from . import _c_quantize as _c_quant
                orig_scale = a_fd_list[0].scale + b_fd.scale
                rounding_code = 0 if ctx.rounding == "ROUND_HALF_EVEN" else 1
                return _c_quant.c_mul_and_quantize_from_arrays(a_arr, b_arr_rep, orig_scale, ctx.scale, rounding_code)
            except Exception:
                # fall through to safe Python path
                pass

        # safe Python path: compute per-element and place into array
        out = array("q")
        prod_scale = a_fd_list[0].scale + b_fd.scale
        for ai in a_fd_list:
            prod = ai.int_value * b_fd.int_value
            out_int = quantize_int(prod, orig_scale=prod_scale, target_scale=ctx.scale, rounding=ctx.rounding)
            out.append(out_int)
        return out

    # both sequences
    a_fd_list = _ensure_fd_list(a_list)
    b_fd_list = _ensure_fd_list(b)

    # fast path predicate
    if _all_pure_integer_and_constant_scale(a_fd_list, b_fd_list):
        try:
            a_arr = array("q", (fd.int_value for fd in a_fd_list))
            b_arr = array("q", (fd.int_value for fd in b_fd_list))
            from . import _c_quantize as _c_quant
            orig_scale = a_fd_list[0].scale + b_fd_list[0].scale
            rounding_code = 0 if ctx.rounding == "ROUND_HALF_EVEN" else 1
            return _c_quant.c_mul_and_quantize_from_arrays(a_arr, b_arr, orig_scale, ctx.scale, rounding_code)
        except Exception:
            # fallback: python path to array
            pass

    # generic safe Python fallback to produce an array('q')
    out = array("q")
    prod_scale = a_fd_list[0].scale + b_fd_list[0].scale
    for ai, bi in zip(a_fd_list, b_fd_list):
        prod = ai.int_value * bi.int_value
        out_int = quantize_int(prod, orig_scale=prod_scale, target_scale=ctx.scale, rounding=ctx.rounding)
        out.append(out_int)
    return out


# -------------------------
# Fast-path predicates & helpers
# -------------------------
def _all_pure_integer_and_constant_scale(a_list: Sequence[FastDecimal], b_list: Sequence[FastDecimal]) -> bool:
    """Return True if all elements in both lists are pure integer-backed and share the same scale."""
    if not a_list or not b_list:
        return False
    first_scale = a_list[0].scale
    for fd in a_list:
        if fd._orig_decimal is not None or fd.scale != first_scale:
            return False
    for fd in b_list:
        if fd._orig_decimal is not None or fd.scale != first_scale:
            return False
    return True


# -------------------------
# Fast integer-array add/sub/mul implementations
# -------------------------

def _mul_ints(a_list: Sequence[FastDecimal], b_list: Sequence[FastDecimal]) -> List[FastDecimal]:
    """
    Use compiled c_mul_and_quantize_from_arrays when available, passing array('q')
    to avoid Python-int overhead. If the C kernel returns a mask marking entries
    that must fall back to Python (overflow/unsafe), only those indices are
    recomputed using the pure-Python `quantize_int`. This gives both speed and
    correctness.
    """
    ctx = get_context()
    n = len(a_list)
    if n == 0:
        return []

    # Extract int values & scales
    a_vals = [fd.int_value for fd in a_list]
    b_vals = [fd.int_value for fd in b_list]
    a_scale = a_list[0].scale
    b_scale = b_list[0].scale
    prod_scale = a_scale + b_scale
    targ_scale = ctx.scale
    rounding = ctx.rounding

    rounding_map = {
        "ROUND_HALF_EVEN": 0,
        "ROUND_HALF_UP": 1,
        "ROUND_HALF_DOWN": 2,
        "ROUND_DOWN": 3,
        "ROUND_UP": 4,
        "ROUND_FLOOR": 5,
        "ROUND_CEILING": 6,
    }
    rounding_code = rounding_map.get(rounding, 0)

    # Try compiled path that expects array('q') to avoid Python boxing
    try:
        a_arr = array("q", a_vals)  # may raise OverflowError if any int doesn't fit 64-bit
        b_arr = array("q", b_vals)
        from . import _c_quantize as _c_quant

        # c kernel expects (a_arr, b_arr, orig_scale, target_scale, rounding_code)
        c_result = _c_quant.c_mul_and_quantize_from_arrays(a_arr, b_arr, prod_scale, targ_scale, rounding_code)

        # Handle both possible return shapes for compatibility:
        # - old: single array('q')
        # - new: (array('q'), array('b'))
        if isinstance(c_result, (tuple, list)):
            res_arr, mask_arr = c_result
        else:
            res_arr = c_result
            mask_arr = None

        # Convert to Python list of ints (fast)
        res_ints = list(res_arr)

        # If no mask, we're done: wrap into FastDecimal objects
        if mask_arr is None:
            return [FastDecimal(int_value=int(x), scale=targ_scale, _orig_decimal=None) for x in res_ints]

        # Otherwise handle masked indices that need Python safe recompute
        mask_list = list(mask_arr)

        # For masked indices, recompute result using Python int path + quantize_int
        q_int = quantize_int
        for i, m in enumerate(mask_list):
            if m:
                prod = a_vals[i] * b_vals[i]
                res_ints[i] = q_int(prod, orig_scale=prod_scale, target_scale=targ_scale, rounding=rounding)

        return [FastDecimal(int_value=int(iv), scale=targ_scale, _orig_decimal=None) for iv in res_ints]

    except Exception:
        # fallback: safe pure-Python path (existing logic)
        prod_ints = [a_vals[i] * b_vals[i] for i in range(n)]
        res_ints = [quantize_int(prod_ints[i], orig_scale=prod_scale, target_scale=targ_scale, rounding=rounding) for i in range(n)]
        return [FastDecimal(int_value=iv, scale=targ_scale, _orig_decimal=None) for iv in res_ints]


def _add_ints(a_list: Sequence[FastDecimal], b_list: Sequence[FastDecimal]) -> List[FastDecimal]:
    """
    Try fast compiled add+quantize via c_add_and_quantize_from_arrays.
    Fallback to Python if conversion/compiled call fails.
    """
    ctx = get_context()
    n = len(a_list)
    if n == 0:
        return []
    a_vals = [fd.int_value for fd in a_list]
    b_vals = [fd.int_value for fd in b_list]
    s_scale = a_list[0].scale
    targ_scale = ctx.scale
    rounding = ctx.rounding

    try:
        a_arr = array("q", a_vals)
        b_arr = array("q", b_vals)
        from . import _c_quantize as _cq
        rounding_code = 0 if rounding == "ROUND_HALF_EVEN" else 1
        prod = _cq.c_add_and_quantize_from_arrays(a_arr, b_arr, s_scale, targ_scale, rounding_code)
        return [FastDecimal(int_value=int(x), scale=targ_scale, _orig_decimal=None) for x in prod]
    except Exception:
        # fallback: pure-Python (existing logic)
        q_int = quantize_int
        if targ_scale == s_scale:
            return [FastDecimal(int_value=(a_vals[i] + b_vals[i]), scale=targ_scale, _orig_decimal=None) for i in range(n)]
        if targ_scale > s_scale:
            factor = 10 ** (targ_scale - s_scale)
            return [FastDecimal(int_value=(a_vals[i] + b_vals[i]) * factor, scale=targ_scale, _orig_decimal=None) for i in range(n)]
        return [FastDecimal(int_value=q_int(a_vals[i] + b_vals[i], orig_scale=s_scale, target_scale=targ_scale, rounding=rounding), scale=targ_scale, _orig_decimal=None) for i in range(n)]


def _sub_ints(a_list: Sequence[FastDecimal], b_list: Sequence[FastDecimal]) -> List[FastDecimal]:
    """
    Try fast compiled sub+quantize via c_sub_and_quantize_from_arrays.
    Fallback to Python if conversion/compiled call fails.
    """
    ctx = get_context()
    n = len(a_list)
    if n == 0:
        return []
    a_vals = [fd.int_value for fd in a_list]
    b_vals = [fd.int_value for fd in b_list]
    s_scale = a_list[0].scale
    targ_scale = ctx.scale
    rounding = ctx.rounding

    try:
        a_arr = array("q", a_vals)
        b_arr = array("q", b_vals)
        from . import _c_quantize as _cq
        rounding_code = 0 if rounding == "ROUND_HALF_EVEN" else 1
        prod = _cq.c_sub_and_quantize_from_arrays(a_arr, b_arr, s_scale, targ_scale, rounding_code)
        return [FastDecimal(int_value=int(x), scale=targ_scale, _orig_decimal=None) for x in prod]
    except Exception:
        # fallback: pure-Python (existing logic)
        q_int = quantize_int
        if targ_scale == s_scale:
            return [FastDecimal(int_value=(a_vals[i] - b_vals[i]), scale=targ_scale, _orig_decimal=None) for i in range(n)]
        if targ_scale > s_scale:
            factor = 10 ** (targ_scale - s_scale)
            return [FastDecimal(int_value=(a_vals[i] - b_vals[i]) * factor, scale=targ_scale, _orig_decimal=None) for i in range(n)]
        return [FastDecimal(int_value=q_int(a_vals[i] - b_vals[i], orig_scale=s_scale, target_scale=targ_scale, rounding=rounding), scale=targ_scale, _orig_decimal=None) for i in range(n)]


# -------------------------
# Public vector ops
# -------------------------

def add_vectors(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> List[FastDecimal]:
    """Elementwise addition (supports scalar broadcast)."""
    # scalar broadcast
    if not _is_sequence(b):
        b_fd = ensure_fastdec(b)
        res: List[FastDecimal] = []
        ctx = get_context()
        for ai in a_list:
            a_fd = ensure_fastdec(ai)
            if a_fd._orig_decimal is None and b_fd._orig_decimal is None and a_fd.scale == ctx.scale and b_fd.scale == ctx.scale:
                res.append(FastDecimal(int_value=a_fd.int_value + b_fd.int_value, scale=ctx.scale, _orig_decimal=None))
            else:
                res.append(a_fd + b_fd)
        return res

    # both sequences
    a_fd_list = [ensure_fastdec(x) for x in a_list]
    b_fd_list = [ensure_fastdec(x) for x in b]
    if _all_pure_integer_and_constant_scale(a_fd_list, b_fd_list):
        return _add_ints(a_fd_list, b_fd_list)

    res: List[FastDecimal] = []
    for a_fd, b_fd in zip(a_fd_list, b_fd_list):
        res.append(a_fd + b_fd)
    return res


def sub_vectors(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> List[FastDecimal]:
    """Elementwise subtraction (supports scalar broadcast)."""
    if not _is_sequence(b):
        b_fd = ensure_fastdec(b)
        res: List[FastDecimal] = []
        ctx = get_context()
        for ai in a_list:
            a_fd = ensure_fastdec(ai)
            if a_fd._orig_decimal is None and b_fd._orig_decimal is None and a_fd.scale == ctx.scale and b_fd.scale == ctx.scale:
                res.append(FastDecimal(int_value=a_fd.int_value - b_fd.int_value, scale=ctx.scale, _orig_decimal=None))
            else:
                res.append(a_fd - b_fd)
        return res

    a_fd_list = [ensure_fastdec(x) for x in a_list]
    b_fd_list = [ensure_fastdec(x) for x in b]
    if _all_pure_integer_and_constant_scale(a_fd_list, b_fd_list):
        return _sub_ints(a_fd_list, b_fd_list)

    res: List[FastDecimal] = []
    for a_fd, b_fd in zip(a_fd_list, b_fd_list):
        res.append(a_fd - b_fd)
    return res


def mul_vectors_frozen(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> List[FastDecimal]:
    """
    Convenience helper: freeze inputs (drop _orig_decimal) and run the fastest integer-backed path.

    - If `b` is a scalar, it will be frozen once.
    - If `b` is a sequence, both lists are frozen then mul_vectors is invoked.
    Use this when you know input values can be safely quantized to the active context.
    """
    from .fastdecimal import FastDecimal as _FD
    # When b is scalar: freeze a_list and b scalar
    if not _is_sequence(b):
        a_frozen = _FD.freeze_list(a_list)
        b_fd = _FD.from_str(str(b)) if not isinstance(b, FastDecimal) else b
        b_frozen = b_fd.freeze()
        return mul_vectors(a_frozen, b_frozen)

    # both sequences: freeze both
    a_frozen = _FD.freeze_list(a_list)
    b_frozen = _FD.freeze_list(b)
    return mul_vectors(a_frozen, b_frozen)


def mul_vectors(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> List[FastDecimal]:
    """Elementwise multiplication (supports scalar broadcast).

    This implementation auto-switches to the fast C-backed path when the active
    DecimalContext.mode requests it:

      - mode == "safe": always use the Python-safe path
      - mode == "fast": always attempt the fast path and raise if it fails
      - mode == "auto": try the fast path, fall back to safe on failure

    The fast path tries to use the integer-array optimized routines when both
    sides are pure integer-backed FastDecimals of matching scale.
    """
    ctx = get_context()
    mode = getattr(ctx, "mode", "safe")

    # Normalize inputs
    a_fd_list = [ensure_fastdec(x) for x in a_list]
    b_is_seq = _is_sequence(b)

    if not b_is_seq:
        b_fd = ensure_fastdec(b)
        b_fd_list = [b_fd] * len(a_fd_list)
    else:
        b_fd_list = [ensure_fastdec(x) for x in b]

    # === SAFE MODE ===
    if mode == "safe":
        return [ai * bi for ai, bi in zip(a_fd_list, b_fd_list)]

    # === FAST / AUTO MODES ===
    try:
        # Use integer-array fast path if all integer-backed and same scale
        if _all_pure_integer_and_constant_scale(a_fd_list, b_fd_list):
            return _mul_ints(a_fd_list, b_fd_list)
        else:
            # fallback: elementwise Python multiply
            return [ai * bi for ai, bi in zip(a_fd_list, b_fd_list)]
    except Exception:
        if mode == "fast":
            raise  # strict fast mode -> propagate error
        # auto mode fallback
        return [ai * bi for ai, bi in zip(a_fd_list, b_fd_list)]


# view/zero-copy helpers
_RAW_VIEW_THRESHOLD = 4096

def mul_vectors_view(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> FastDecimalView:
    """
    Fast zero-copy multiply that returns a FastDecimalView (array('q')).
    - Requires inputs to be suitable for the raw compiled path (pure integer-backed
      or convertible).
    - Falls back to Python-produced array('q') when strict fast path isn't possible.
    """
    ctx = get_context()
    arr = mul_vectors_raw(a_list, b)  # returns array('q')
    return FastDecimalView(arr, ctx.scale)


def mul_vectors_fast(a_list: Sequence[Scalar], b: Union[Sequence[Scalar], Scalar]) -> List[FastDecimal]:
    """
    Ultra-fast vector multiply path with overflow-safe fallback.

    Uses C-optimized quantize kernel and Python fallback for overflowed elements.
    """
    from . import _c_quantize
    ctx = get_context()

    # Normalize inputs
    if not _is_sequence(b):
        b = [b] * len(a_list)

    a_fd_list = [ensure_fastdec(x) for x in a_list]
    b_fd_list = [ensure_fastdec(x) for x in b]

    if not _all_pure_integer_and_constant_scale(a_fd_list, b_fd_list):
        return mul_vectors(a_list, b)

    a_arr = array("q", [fd.int_value for fd in a_fd_list])
    b_arr = array("q", [fd.int_value for fd in b_fd_list])

    try:
        prod_scale = a_fd_list[0].scale + b_fd_list[0].scale
        rounding_code = 1 if ctx.rounding == "ROUND_HALF_UP" else 0
        c_result = _c_quantize.c_mul_and_quantize_from_arrays(a_arr, b_arr, prod_scale, ctx.scale, rounding_code)

        if isinstance(c_result, (tuple, list)):
            res_arr, mask_arr = c_result
        else:
            res_arr = c_result
            mask_arr = None

        # handle fallback for overflowed indices
        if mask_arr is not None and any(mask_arr):
            mask_list = list(mask_arr)
            for i, m in enumerate(mask_list):
                if m:
                    fallback_val = (a_fd_list[i] * b_fd_list[i]).quantize(ctx.scale)
                    res_arr[i] = fallback_val.int_value

    except Exception:
        return mul_vectors(a_list, b)

    return [FastDecimal(int_value=int(x), scale=ctx.scale, _orig_decimal=None) for x in res_arr]

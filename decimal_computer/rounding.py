# decimal_computer/rounding.py
"""
Integer-based rounding / quantize utilities.

Functions here operate purely on integer representations:
 - `quantize_int(int_value, orig_scale, target_scale, rounding)`
   takes an integer representing value * 10**orig_scale, and returns an
   integer representing value * 10**target_scale applying the requested rounding.

Supported rounding modes (strings):
 - "ROUND_HALF_EVEN"
 - "ROUND_HALF_UP"
 - "ROUND_DOWN"      (truncate toward zero)
 - "ROUND_FLOOR"     (toward -inf)
 - "ROUND_CEILING"   (toward +inf)
 - "ROUND_HALF_DOWN" (tie goes down)
 - "ROUND_UP"        (away from zero)

The algorithms avoid using Decimal and rely on integer math for determinism and speed.
"""
from __future__ import annotations
from typing import Literal

RoundingMode = Literal[
    "ROUND_HALF_EVEN",
    "ROUND_HALF_UP",
    "ROUND_DOWN",
    "ROUND_FLOOR",
    "ROUND_CEILING",
    "ROUND_HALF_DOWN",
    "ROUND_UP",
]


def _signed_divmod_abs(a: int, b: int) -> tuple[int, int, int]:
    """
    Return (quotient, remainder, sign) where quotient = trunc(a/b) toward zero,
    remainder = abs(a) % b, sign = 1 if a>=0 else -1.
    """
    sign = 1 if a >= 0 else -1
    abs_a = a if a >= 0 else -a
    q, r = divmod(abs_a, b)
    return q, r, sign


def quantize_int(int_value: int, orig_scale: int, target_scale: int, rounding: RoundingMode) -> int:
    """
    Quantize integer `int_value` from scale `orig_scale` -> `target_scale` using rounding.

    If target_scale >= orig_scale -> we need to increase scale (multiply by 10**delta).
    If target_scale < orig_scale -> we reduce precision (divide by factor) and round.

    Returns the integer value at target_scale.
    """
    if orig_scale == target_scale:
        return int_value

    if target_scale > orig_scale:
        # increase scale (add trailing zeros)
        factor = 10 ** (target_scale - orig_scale)
        return int_value * factor

    # target_scale < orig_scale -> we must divide and round
    factor = 10 ** (orig_scale - target_scale)
    q, r, sign = _signed_divmod_abs(int_value, factor)  # q is >=0
    # r in [0, factor-1]

    # Quick mapping for rounding decisions:
    if rounding == "ROUND_DOWN":
        # truncate toward zero
        return q * sign

    if rounding == "ROUND_UP":
        # away from zero: increment q if remainder != 0
        if r != 0:
            return (q + 1) * sign
        return q * sign

    if rounding == "ROUND_FLOOR":
        # toward -inf
        if sign < 0 and r != 0:
            return (q + 1) * -1
        return q * sign

    if rounding == "ROUND_CEILING":
        # toward +inf
        if sign > 0 and r != 0:
            return (q + 1) * sign
        return q * sign

    # half decisions: compare r*2 with factor
    cmp = r * 2 - factor  # <0 => less than half, ==0 => tie, >0 => greater than half

    if rounding == "ROUND_HALF_UP":
        if cmp >= 0:
            return (q + 1) * sign
        return q * sign

    if rounding == "ROUND_HALF_DOWN":
        if cmp > 0:
            return (q + 1) * sign
        return q * sign

    if rounding == "ROUND_HALF_EVEN":
        if cmp > 0:
            return (q + 1) * sign
        if cmp < 0:
            return q * sign
        # tie: choose even (q is abs quotient)
        if (q % 2) == 0:
            return q * sign
        return (q + 1) * sign

    # fallback: raise for unsupported mode
    raise ValueError(f"Unsupported rounding mode: {rounding}")

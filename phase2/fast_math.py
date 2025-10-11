# phase2/fast_math.py
"""
Simple fast_math helpers used as small backends in tests.
They accept either integers or lists-of-decimal-digits (MSB -> LSB).
For lists, we convert to integers, do the operation, and convert back
to a digit list to keep the API convenient for tests.
"""

from __future__ import annotations
from typing import List, Union

NumberOrDigits = Union[int, List[int]]

def _digits_to_int(d: List[int]) -> int:
    """Convert list-of-digits (MSB->LSB) to int."""
    if not d:
        return 0
    val = 0
    for digit in d:
        val = val * 10 + int(digit)
    return val

def _int_to_digits(n: int) -> List[int]:
    """Convert non-negative int to list-of-digits (MSB->LSB)."""
    if n == 0:
        return [0]
    if n < 0:
        n = -n  # tests use positive values; negative handling not required here
    s = str(n)
    return [int(ch) for ch in s]

def digit_mul(a: NumberOrDigits, b: NumberOrDigits) -> NumberOrDigits:
    """
    Multiply either two ints or two digit-lists.
    If inputs are lists, returns a list of digits (MSB->LSB).
    """
    # both lists -> operate on them
    if isinstance(a, list) and isinstance(b, list):
        ai = _digits_to_int(a)
        bi = _digits_to_int(b)
        prod = ai * bi
        return _int_to_digits(prod)
    # one list and one int -> normalize to ints and return int (tests don't use this)
    if isinstance(a, list) and isinstance(b, int):
        return _digits_to_int(a) * b
    if isinstance(b, list) and isinstance(a, int):
        return a * _digits_to_int(b)
    # fallback: numeric multiply
    return int(a) * int(b)

def digit_add(a: NumberOrDigits, b: NumberOrDigits) -> NumberOrDigits:
    """
    Add either two ints or two digit-lists.
    If inputs are lists, returns a list of digits (MSB->LSB).
    """
    if isinstance(a, list) and isinstance(b, list):
        ai = _digits_to_int(a)
        bi = _digits_to_int(b)
        s = ai + bi
        return _int_to_digits(s)
    if isinstance(a, list) and isinstance(b, int):
        return _digits_to_int(a) + int(b)
    if isinstance(b, list) and isinstance(a, int):
        return int(a) + _digits_to_int(b)
    return int(a) + int(b)

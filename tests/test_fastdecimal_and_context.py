# tests/test_fastdecimal_and_context.py
"""
Pytest tests for decimal_computer.context and decimal_computer.fastdecimal.

Run with:
    pytest -q

These tests exercise:
- context defaulting and override behavior
- construction of FastDecimal from string/float/int
- addition, subtraction, multiplication, division with rounding/scale behavior
"""

import pytest
from decimal import Decimal

from decimal_computer.context import DecimalContext, get_context
from decimal_computer.fastdecimal import FastDecimal


def test_default_context():
    # Without any explicit context, get_context returns a default one
    ctx = get_context()
    assert isinstance(ctx, DecimalContext)
    assert ctx.precision >= 1
    assert ctx.scale >= 0


def test_context_nesting_and_use():
    base = get_context()
    with DecimalContext(precision=10, scale=4, rounding='ROUND_HALF_UP') as ctx:
        assert get_context().scale == 4
        assert get_context().precision == 10
    # after exit, previous context should be restored (or replaced by default)
    assert get_context().precision == base.precision


def test_fastdecimal_from_and_to_str():
    with DecimalContext(precision=10, scale=3, rounding='ROUND_HALF_EVEN'):
        a = FastDecimal.from_str("12.34567")  # will round to scale=3
        assert isinstance(a, FastDecimal)
        assert str(a) == "12.346"  # rounded half-even: .34567 -> .346


def test_add_sub_same_context():
    with DecimalContext(precision=10, scale=2, rounding='ROUND_HALF_UP'):
        a = FastDecimal.from_str("1.25")
        b = FastDecimal.from_str("2.34")
        assert str(a + b) == "3.59"
        assert str(b - a) == "1.09"


def test_mul_respects_context_scale_and_rounding():
    with DecimalContext(precision=12, scale=4, rounding='ROUND_HALF_UP'):
        a = FastDecimal.from_str("1.23456")
        b = FastDecimal.from_str("2.5")
        prod = a * b
        # using Decimal for expected value with the same quantize
        expected = (Decimal("1.23456") * Decimal("2.5")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        assert str(prod) == format(expected, 'f')


def test_division_and_zero_guard():
    with DecimalContext(precision=12, scale=4, rounding='ROUND_HALF_EVEN'):
        a = FastDecimal.from_str("12.0")
        b = FastDecimal.from_str("4.0")
        assert str(a / b) == "3"
        c = FastDecimal.from_int(0)
        with pytest.raises(ZeroDivisionError):
            _ = a / c


def test_comparisons_and_mixed_inputs():
    with DecimalContext(precision=10, scale=3, rounding='ROUND_HALF_EVEN'):
        a = FastDecimal.from_str("1.000")
        b = FastDecimal.from_float(1.0)
        c = FastDecimal.from_str("1.001")
        assert a == b
        assert a < c
        # mixed python numeric input
        assert (a + 1) == FastDecimal.from_str("2.000")

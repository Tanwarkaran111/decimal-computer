# tests/test_vector_ops.py
import pytest
from decimal import Decimal

from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops


def _fmt_decimal_like_fastdecimal(d: Decimal) -> str:
    """Format Decimal same way FastDecimal.__str__() does (strip trailing zeros)."""
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def test_add_vectors_basic_and_broadcast():
    with DecimalContext(precision=10, scale=2, rounding="ROUND_HALF_EVEN"):
        a = [FastDecimal.from_int(1), FastDecimal.from_int(2), FastDecimal.from_int(3)]
        # add scalar
        out = vector_ops.add_vectors(a, 1)
        assert [str(x) for x in out] == ["2", "3", "4"]

        # add list
        b = [FastDecimal.from_int(4), FastDecimal.from_int(5), FastDecimal.from_int(6)]
        out2 = vector_ops.add_vectors(a, b)
        assert [str(x) for x in out2] == ["5", "7", "9"]


def test_mul_vectors_integer_path():
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a = [FastDecimal.from_str("1.2345"), FastDecimal.from_str("2.0")]
        b = [FastDecimal.from_str("2.0"), FastDecimal.from_str("3.0")]
        out = vector_ops.mul_vectors(a, b)
        # expected using Decimal with same context, but format to match FastDecimal.__str__()
        expected0 = (Decimal("1.2345") * Decimal("2.0")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        expected1 = (Decimal("2.0") * Decimal("3.0")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        assert str(out[0]) == _fmt_decimal_like_fastdecimal(expected0)
        assert str(out[1]) == _fmt_decimal_like_fastdecimal(expected1)


def test_sub_vectors_mixed_inputs_and_lengths():
    with DecimalContext(precision=10, scale=3, rounding="ROUND_HALF_EVEN"):
        a = [1.234, "2.000", 3]
        b = 1
        out = vector_ops.sub_vectors(a, b)
        assert [str(x) for x in out] == ["0.234", "1", "2"]

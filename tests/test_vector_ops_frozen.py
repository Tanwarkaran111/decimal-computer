# tests/test_vector_ops_frozen.py
from decimal import Decimal
from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops

def test_mul_vectors_frozen_basic():
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a = [FastDecimal.from_str("1.2345"), FastDecimal.from_str("2.0")]
        b = [FastDecimal.from_str("2.0"), FastDecimal.from_str("3.0")]
        out = vector_ops.mul_vectors_frozen(a, b)
        expected0 = (Decimal("1.2345") * Decimal("2.0")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        expected1 = (Decimal("2.0") * Decimal("3.0")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        assert str(out[0]) == format(expected0, 'f').rstrip('0').rstrip('.')
        assert str(out[1]) == format(expected1, 'f').rstrip('0').rstrip('.')

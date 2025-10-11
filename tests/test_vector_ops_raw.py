# tests/test_vector_ops_raw.py
from decimal import Decimal
from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer import vector_ops

def test_mul_vectors_raw_basic():
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a = [FastDecimal.from_str("1.2345"), FastDecimal.from_str("2.0")]
        b = [FastDecimal.from_str("2.0"), FastDecimal.from_str("3.0")]
        out_arr = vector_ops.mul_vectors_raw(a, b)
        # convert first element to Decimal for comparison
        expected0 = (Decimal("1.2345") * Decimal("2.0")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        assert str(expected0) == str(Decimal(out_arr[0]).scaleb(-4))

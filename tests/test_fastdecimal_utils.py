# tests/test_fastdecimal_utils.py
import pytest
from decimal import Decimal

from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal


def test_to_float_and_neg_abs_hash_and_json():
    with DecimalContext(precision=10, scale=3, rounding="ROUND_HALF_EVEN"):
        a = FastDecimal.from_str("12.345")
        assert isinstance(a.to_float(), float)
        assert float(a.to_decimal()) == a.to_float()

        # negation
        neg = -a
        assert str(neg) == "-12.345"
        assert str(abs(neg)) == "12.345"

        # hash stable and equal for equivalent values
        b = FastDecimal.from_str("12.345")
        assert hash(a) == hash(b)
        assert a == b

        # serialization roundtrip
        s = a.to_json()
        assert "int" in s and "scale" in s
        a2 = FastDecimal.from_json(s)
        assert isinstance(a2, FastDecimal)
        assert a2.int_value == a.int_value
        assert a2.scale == a.scale


def test_hash_in_collections():
    with DecimalContext(precision=8, scale=2):
        items = {FastDecimal.from_str("1.23"): "x", FastDecimal.from_str("2.00"): "y"}
        assert items[FastDecimal.from_str("1.23")] == "x"
        assert FastDecimal.from_str("2.00") in items

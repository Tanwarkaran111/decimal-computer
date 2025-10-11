# tests/test_scale_alignment.py
from decimal import Decimal

from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal

def _fmt_decimal_like_fastdecimal(d: Decimal) -> str:
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def test_alignment_integer_backed_fast_path():
    # Create two FastDecimals under different creation contexts (different stored scales)
    with DecimalContext(precision=10, scale=2, rounding="ROUND_HALF_EVEN"):
        a = FastDecimal.from_str("12.34")  # stored at scale 2
    with DecimalContext(precision=10, scale=4, rounding="ROUND_HALF_EVEN"):
        b = FastDecimal.from_str("0.0056")  # stored at scale 4

    # Now perform arithmetic under target context scale=3
    with DecimalContext(precision=12, scale=3, rounding="ROUND_HALF_UP"):
        res = a + b
        # expected via Decimal quantize to same target
        expected = (Decimal("12.34") + Decimal("0.0056")).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
        assert str(res) == _fmt_decimal_like_fastdecimal(expected)

def test_alignment_mixed_fallback():
    # If one value preserves orig_decimal, ensure fallback still matches
    with DecimalContext(precision=10, scale=2, rounding="ROUND_HALF_EVEN"):
        a = FastDecimal.from_str("1.23")
    # create b but keep orig_decimal by constructing under same context (from_float/from_str both set orig)
    with DecimalContext(precision=10, scale=4, rounding="ROUND_HALF_EVEN"):
        b = FastDecimal.from_float(0.0056)  # keeps orig_decimal
    with DecimalContext(precision=12, scale=3, rounding="ROUND_HALF_UP"):
        res = a + b
        expected = (Decimal("1.23") + Decimal("0.0056")).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
        assert str(res) == _fmt_decimal_like_fastdecimal(expected)

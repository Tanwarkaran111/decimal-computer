# tests/test_freeze.py
from decimal import Decimal
from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal

def test_freeze_single_and_list():
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a = FastDecimal.from_str("1.23456")   # has _orig_decimal
        af = a.freeze()
        assert af._orig_decimal is None
        # numeric equality when printed matches Decimal quantize
        expected = Decimal("1.23456").quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        assert str(af) == format(expected, 'f').rstrip('0').rstrip('.')

        lst = [FastDecimal.from_str("1.2345"), FastDecimal.from_str("2.5")]
        fl = FastDecimal.freeze_list(lst)
        assert all(x._orig_decimal is None for x in fl)
        # check lengths and sample values
        assert len(fl) == 2
        assert str(fl[0]) == "1.2345"

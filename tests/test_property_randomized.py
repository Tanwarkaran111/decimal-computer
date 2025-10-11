# tests/test_property_randomized.py
"""
Deterministic randomized property tests comparing FastDecimal behaviour with
decimal.Decimal for many random values and contexts.

Handles the '-0' vs '0' formatting difference gracefully.
"""
from decimal import Decimal, localcontext
import random
import pytest

from decimal_computer.context import DecimalContext
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer.rounding import quantize_int

RNG_SEED = 54321

def _fmt_decimal_like_fastdecimal(d: Decimal) -> str:
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    # normalize '-0' to '0' (FastDecimal prints '0')
    if s in ("-0", "-0.0"):
        s = "0"
    return s

@pytest.mark.parametrize("trials", [1000])
def test_random_add_mul_quantize_equivalence(trials):
    random.seed(RNG_SEED)
    contexts = [
        {"precision": 12, "scale": 2, "rounding": "ROUND_HALF_UP"},
        {"precision": 12, "scale": 3, "rounding": "ROUND_HALF_EVEN"},
        {"precision": 10, "scale": 4, "rounding": "ROUND_HALF_DOWN"},
    ]

    for ctxcfg in contexts:
        with DecimalContext(**ctxcfg):
            safe_prec = max(50, ctxcfg["precision"] * 4)
            for _ in range(trials):
                int_part = random.randint(-10000, 10000)
                frac_digits = random.randint(0, 6)
                frac_int = random.randint(0, 10**frac_digits - 1) if frac_digits > 0 else 0
                if frac_digits == 0:
                    s1 = str(int_part)
                else:
                    sign = "-" if int_part < 0 else ""
                    abs_int = abs(int_part)
                    s1 = f"{sign}{abs_int}.{str(frac_int).zfill(frac_digits)}"

                if random.random() < 0.6:
                    s2 = str(random.randint(-50, 50))
                else:
                    fd2 = random.randint(0, 4)
                    frac2 = random.randint(0, 10**fd2 - 1) if fd2 > 0 else 0
                    s2 = f"{random.randint(-50, 50)}.{str(frac2).zfill(fd2)}"

                a = FastDecimal.from_str(s1)
                b = FastDecimal.from_str(s2)

                fd_add = a + b
                fd_mul = a * b

                if a._orig_decimal is not None or b._orig_decimal is not None:
                    dec_a = a._orig_decimal if a._orig_decimal is not None else Decimal(a.int_value).scaleb(-a.scale)
                    dec_b = b._orig_decimal if b._orig_decimal is not None else Decimal(b.int_value).scaleb(-b.scale)
                else:
                    dec_a = Decimal(a.int_value).scaleb(-a.scale)
                    dec_b = Decimal(b.int_value).scaleb(-b.scale)

                from decimal_computer.context import get_context as _get_ctx
                act_ctx = _get_ctx()
                quant = Decimal(1).scaleb(-act_ctx.scale)

                with localcontext() as lc:
                    lc.prec = safe_prec
                    qa = dec_a.quantize(quant, rounding=act_ctx.rounding)
                    qb = dec_b.quantize(quant, rounding=act_ctx.rounding)
                    a_int = int(qa.scaleb(act_ctx.scale).to_integral_value(rounding=act_ctx.rounding))
                    b_int = int(qb.scaleb(act_ctx.scale).to_integral_value(rounding=act_ctx.rounding))
                    add_int = a_int + b_int
                    dec_expected_add = Decimal(add_int).scaleb(-act_ctx.scale)
                    dec_expected_mul = (dec_a * dec_b).quantize(quant, rounding=act_ctx.rounding)

                exp_add = _fmt_decimal_like_fastdecimal(dec_expected_add)
                exp_mul = _fmt_decimal_like_fastdecimal(dec_expected_mul)

                assert str(fd_add) == exp_add
                assert str(fd_mul) == exp_mul

@pytest.mark.parametrize("trials", [200])
def test_random_quantize_int_matches_decimal(trials):
    random.seed(RNG_SEED + 1)
    scales = [0, 1, 2, 3, 4, 6, 8]
    rounding_modes = [
        "ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_HALF_DOWN",
        "ROUND_DOWN", "ROUND_UP", "ROUND_FLOOR", "ROUND_CEILING"
    ]

    for _ in range(trials):
        orig_scale = random.choice(scales)
        target_scale = random.choice(scales)
        magnitude = random.randint(0, 10**6)
        sign = -1 if random.random() < 0.2 else 1
        int_value = sign * magnitude
        rounding = random.choice(rounding_modes)

        with localcontext() as lc:
            lc.prec = 50
            dec = Decimal(int_value).scaleb(-orig_scale)
            quant = Decimal(1).scaleb(-target_scale)
            dec_q = dec.quantize(quant, rounding=rounding)
            dec_res_int = int(dec_q.scaleb(target_scale).to_integral_value(rounding=rounding))

        int_res = quantize_int(int_value, orig_scale=orig_scale, target_scale=target_scale, rounding=rounding)
        assert int_res == dec_res_int

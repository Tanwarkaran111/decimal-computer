# tests/test_rounding.py
import pytest

from decimal_computer.rounding import quantize_int

def test_increase_scale_noop():
    assert quantize_int(1234, orig_scale=2, target_scale=4, rounding="ROUND_HALF_EVEN") == 123400

def test_round_down_truncate_positive():
    # 12345 at orig_scale=3 => value 12.345 -> target_scale=2 => 12.34 truncated
    assert quantize_int(12345, orig_scale=3, target_scale=2, rounding="ROUND_DOWN") == 1234

def test_round_half_up_positive():
    # 1.235 (1235 at scale=3) -> scale=2 should round to 1.24
    assert quantize_int(1235, orig_scale=3, target_scale=2, rounding="ROUND_HALF_UP") == 124

def test_round_half_even_tie_behavior_examples():
    # Craft exact tie examples:
    # value = 1.255 with orig_scale=3 -> 1255 ; target_scale=2 factor=10 -> q=125 r=5 (tie)
    # q=125 odd -> should round up to 126 (because q odd)
    assert quantize_int(1255, orig_scale=3, target_scale=2, rounding="ROUND_HALF_EVEN") == 126
    # if q even:
    # value = 1.245 with orig_scale=3 -> 1245 -> q=124 even -> stays 124
    assert quantize_int(1245, orig_scale=3, target_scale=2, rounding="ROUND_HALF_EVEN") == 124

def test_round_half_down_tie():
    # half-down: tie should round down
    assert quantize_int(1255, orig_scale=3, target_scale=2, rounding="ROUND_HALF_DOWN") == 125

def test_round_floor_negative():
    # -1.235 -> floor to 2 decimals -> -1.24 -> -124
    assert quantize_int(-1235, orig_scale=3, target_scale=2, rounding="ROUND_FLOOR") == -124

def test_round_ceiling_positive():
    # 1.231 -> ceiling to 2 decimals -> 1.24
    assert quantize_int(1231, orig_scale=3, target_scale=2, rounding="ROUND_CEILING") == 124

def test_round_up_away_from_zero():
    assert quantize_int(-1201, orig_scale=3, target_scale=2, rounding="ROUND_UP") == -121
    assert quantize_int(1201, orig_scale=3, target_scale=2, rounding="ROUND_UP") == 121

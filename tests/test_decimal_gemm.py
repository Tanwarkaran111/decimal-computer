from decimal_computer.decimal_digit_starter import int_to_digits, digits_to_int, digit_mul
from decimal_computer.decimal_gemm import int_mul_via_digits, decimal_gemm_naive

def test_digit_roundtrip():
    assert digits_to_int(int_to_digits(12345)) == 12345

def test_digit_mul():
    A = int_to_digits(12)
    B = int_to_digits(34)
    assert digits_to_int(digit_mul(A, B)) == 408

def test_int_mul_via_digits():
    assert int_mul_via_digits(123, 45) == 5535

def test_decimal_gemm():
    A = [[12, 3], [4, 5]]
    B = [[2, 1], [10, 2]]
    assert decimal_gemm_naive(A, B) == [[54, 18], [58, 14]]
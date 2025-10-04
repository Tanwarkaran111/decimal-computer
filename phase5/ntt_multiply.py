"""
phase5/ntt_multiply.py

Number Theoretic Transform (NTT) based exact convolution & integer multiplication.

This module implements:
- ntt(a, invert, mod, root) : in-place Cooley-Tukey NTT (length must be a power of two).
- convolution_mod(a, b, mod, root) : circular convolution modulo `mod`.
- multiply_polynomials_mod(a, b) : multiply integer-coefficient polynomials exactly using NTT
  over a chosen prime modulus (useful for modular arithmetic applications).
- multiply_ints_ntt(x, y, base=1000) : multiply two integers exactly by splitting into base
  limbs, performing convolution under multiple NTT-friendly primes, and reconstructing via CRT.

Notes:
- This is a self-contained, pure-Python implementation meant for correctness and education.
- For production you may prefer to use GMP / specialized libraries, but NTT provides
  exact O(n log n) multiplication without floating point rounding.

Primarily tested with moderate sizes; choose `base` to control limb count.

"""

import math
from typing import List, Sequence, Tuple

# ---- Useful NTT-friendly primes (of form k*2^m + 1) with known primitive root
# Common choices (sufficient for many sizes):
# (mod, primitive_root)
_NTT_PRIMES = [
    (167772161, 3),      # 5 * 2^25 + 1
    (469762049, 3),      # 7 * 2^26 + 1
    (1224736769, 3),     # 73 * 2^24 + 1
]
# Product of these three primes > 2**90 (very large), enough for typical reconstructions.

# ----------------------
# NTT core (iterative, inplace)
# ----------------------

def _ntt(a: List[int], invert: bool, mod: int, root: int) -> None:
    """In-place Cooley-Tukey iterative NTT. Length must be power of two.
    If invert is True, computes inverse NTT (divides by n at end).
    """
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        # compute primitive length-th root
        # wlen = root^{(mod-1)/length}
        wlen = pow(root, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        for i in range(0, n, length):
            w = 1
            half = length >> 1
            for k in range(i, i + half):
                u = a[k]
                v = (a[k + half] * w) % mod
                a[k] = (u + v) % mod
                a[k + half] = (u - v) % mod
                w = (w * wlen) % mod
        length <<= 1

    if invert:
        inv_n = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = (a[i] * inv_n) % mod


def _next_power_of_two(n: int) -> int:
    return 1 << ((n - 1).bit_length())

# ----------------------
# Modular convolution using NTT
# ----------------------

def convolution_mod(a: Sequence[int], b: Sequence[int], mod: int, root: int) -> List[int]:
    """Convolve two integer sequences modulo `mod` using NTT. Returns length n+m-1 result.
    Values in a,b must be already reduced modulo `mod` if needed.
    """
    na = len(a)
    nb = len(b)
    if na == 0 or nb == 0:
        return []
    n = _next_power_of_two(na + nb - 1)
    A = list(a) + [0] * (n - na)
    B = list(b) + [0] * (n - nb)
    _ntt(A, False, mod, root)
    _ntt(B, False, mod, root)
    for i in range(n):
        A[i] = (A[i] * B[i]) % mod
    _ntt(A, True, mod, root)
    # trim to required length
    return [A[i] for i in range(na + nb - 1)]

# ----------------------
# CRT (Garner) reconstruction for list of residues and moduli
# ----------------------

def _crt_garner(residues: List[int], mods: List[int]) -> int:
    """Reconstruct unique value x modulo prod(mods) from residues using Garner's algorithm.
    Returns integer in [0, M-1] where M = product(mods).
    """
    assert len(residues) == len(mods)
    k = len(mods)
    # coeffs: c[i] will hold the representation
    coeffs = [0] * k
    mods_prod = [1] * k
    for i in range(k):
        coeffs[i] = residues[i]
        for j in range(i):
            # compute coeffs[i] = (coeffs[i] - coeffs[j]) * inv(mods_prod[j]) mod mods[i]
            inv = pow(mods_prod[j] % mods[i], mods[i] - 2, mods[i])
            coeffs[i] = (coeffs[i] - coeffs[j]) * inv % mods[i]
        mods_prod[i] = 1
        for j in range(i + 1):
            mods_prod[i] *= mods[j]
    # reconstruct final integer
    x = 0
    mult = 1
    for i in range(k):
        x += coeffs[i] * mult
        mult *= mods[i]
    return x

# Simpler CRT for two moduli (faster path)
def _crt_two(r1: int, m1: int, r2: int, m2: int) -> int:
    # returns x congruent to r1 mod m1 and r2 mod m2, x in [0, m1*m2)
    inv = pow(m1, -1, m2)
    t = (r2 - r1) * inv % m2
    return r1 + m1 * t

# ----------------------
# Multiply polynomials modulo arbitrary modulus using one NTT prime
# ----------------------

def multiply_polynomials_mod(a: Sequence[int], b: Sequence[int], mod: int = 998244353, root: int = 3) -> List[int]:
    # reduce coefficients
    a_red = [x % mod for x in a]
    b_red = [x % mod for x in b]
    return convolution_mod(a_red, b_red, mod, root)

# ----------------------
# Exact integer multiplication using multiple NTT primes + CRT
# ----------------------

def multiply_ints_ntt(x: int, y: int, base: int = 1000) -> int:
    """Multiply two integers exactly using NTT convolution + CRT across several primes.
    - base: decimal base to split digits (e.g. 1000 -> 3 decimal digits per limb)

    Algorithm overview:
      1. Split |x| and |y| into limb arrays (least-significant first) in given base.
      2. For each NTT prime, compute convolution modulo that prime.
      3. Reconstruct convolution integer coefficients via CRT (Garner) to get full integer coefficients.
      4. Handle carries in `base` and recombine to final integer.

    This yields an exact product so long as the product of chosen primes exceeds the maximum
    possible coefficient magnitude; the provided prime set is large for practical sizes.
    """
    if x == 0 or y == 0:
        return 0
    sign = -1 if (x < 0) ^ (y < 0) else 1
    ax = abs(x)
    ay = abs(y)

    def to_digits(n: int):
        if n == 0:
            return [0]
        digs = []
        while n:
            digs.append(n % base)
            n //= base
        return digs

    da = to_digits(ax)
    db = to_digits(ay)
    target_len = len(da) + len(db) - 1

    residues_per_prime = []  # list of lists
    mods = []
    for (mod, root) in _NTT_PRIMES:
        mods.append(mod)
        conv_mod = convolution_mod(da, db, mod, root)
        # ensure length
        conv_mod += [0] * (target_len - len(conv_mod))
        residues_per_prime.append(conv_mod)

    # Reconstruct each coefficient via CRT
    coeffs = []
    k = len(mods)
    for i in range(target_len):
        residues = [residues_per_prime[p][i] for p in range(k)]
        val = _crt_garner(residues, mods)
        coeffs.append(val)

    # Handle carries in base
    carry = 0
    for i in range(len(coeffs)):
        total = coeffs[i] + carry
        carry = total // base
        coeffs[i] = total % base
    while carry:
        coeffs.append(carry % base)
        carry //= base

    # reconstruct integer
    result = 0
    for d in reversed(coeffs):
        result = result * base + d

    return sign * result

# ----------------------
# Self-test harness
# ----------------------

def _self_test():
    import random, sys
    print("NTT primes:", _NTT_PRIMES)
    # simple polynomial test
    p1 = [1, 2, 3]
    p2 = [4, 5]
    prod = multiply_polynomials_mod(p1, p2, mod=_NTT_PRIMES[0][0], root=_NTT_PRIMES[0][1])
    print("polys", p1, "*", p2, "=", prod, "(expected [4,13,22,15])")

    tests = [
        (0, 12345),
        (123, 456),
        (99999, 0),
        (123456789, 987654321),
    ]
    for a, b in tests:
        got = multiply_ints_ntt(a, b)
        print(f"{a} * {b} = {got} (expected {a*b})")
        if got != a * b:
            print("ERROR: mismatch!")
            sys.exit(1)

    # random stress (smaller)
    for _ in range(20):
        a = random.randint(-10**30, 10**30)
        b = random.randint(-10**30, 10**30)
        if multiply_ints_ntt(a, b) != a * b:
            print("ERROR on random test")
            sys.exit(1)
    print("Random tests passed.")

    # larger test
    a = 10**200 + 12345678901234567890
    b = 10**150 + 9876543210987654321
    print("Large multiply check (may be slow)...")
    ok = multiply_ints_ntt(a, b) == a * b
    print("Large multiply pass?", ok)


if __name__ == '__main__':
    _self_test()

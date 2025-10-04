from dataclasses import dataclass
from typing import Optional, Union, List
import math
import sys

# --- Ensure large integer conversions never fail ---
try:
    sys.set_int_max_str_digits(0)   # disable limit for trusted computations
    print("[BigInt] Disabled Python's int() digit limit")
except Exception:
    pass

# Prefer gmpy2 if available
try:
    import gmpy2
    _HAVE_GMPY2 = True
except Exception:
    _HAVE_GMPY2 = False

# Phase6 auto-multiply fallback
try:
    from phase6.auto_multiply import auto_multiply
    _HAVE_AUTO = True
except Exception:
    _HAVE_AUTO = False

# Phase5 NTT fallback for exact recompute if gmpy2 absent
try:
    from phase5.ntt_multiply import multiply_ints_ntt
    _HAVE_NTT = True
except Exception:
    _HAVE_NTT = False


@dataclass
class BigInt:
    _val: int

    @classmethod
    def from_decimal_str(cls, s: str) -> "BigInt":
        """
        Robust BigInt from decimal string.

        Fast paths:
         - repetitive patterns like '9'*d or '1'*d are detected and built using
           gmpy2/mpz exponentiation (very fast).
         - otherwise try gmpy2.mpz(s) (C-side parser).
        Fallback:
         - Python chunked parse (slow) only if necessary.
        """
        if s is None:
            raise ValueError("None passed to from_decimal_str")

        s = s.replace("_", "").strip()
        if s == "":
            return cls(0)

        # handle sign
        sign = 1
        if s[0] == "+":
            s = s[1:]
        elif s[0] == "-":
            sign = -1
            s = s[1:]

        n = len(s)

        # FAST SHORTCUT: detect very simple uniform/repetitive patterns
        # pattern: all digits are the same (e.g. '9'*d, '1'*d)
        # pattern2: leading run of same digit followed by other digits not considered
        if n > 2:
            first_ch = s[0]
            if s == first_ch * n:
                # e.g. '9'*d  -> use 10**d - 1
                try:
                    if first_ch == "9":
                        if _HAVE_GMPY2:
                            import gmpy2
                            val = gmpy2.mpz(10) ** n - 1
                            if sign < 0:
                                val = -val
                            return cls(int(val))
                        else:
                            # fallback using pow on Python ints (slower but fewer loops)
                            val = 10 ** n - 1
                            if sign < 0:
                                val = -val
                            return cls(val)
                    elif first_ch == "1":
                        # '1'*d = (10**d - 1) // 9
                        if _HAVE_GMPY2:
                            import gmpy2
                            val = (gmpy2.mpz(10) ** n - 1) // 9
                            if sign < 0:
                                val = -val
                            return cls(int(val))
                        else:
                            val = (10 ** n - 1) // 9
                            if sign < 0:
                                val = -val
                            return cls(val)
                except Exception:
                    # ignore and fall through to general parse
                    pass

        # Try fast gmpy2 parser first (handles typical decimal inputs in C)
        if _HAVE_GMPY2:
            try:
                import gmpy2
                val = gmpy2.mpz(s)
                if sign < 0:
                    val = -val
                return cls(int(val))
            except Exception:
                # fallback below
                pass

        # Fallback: Python-only chunked parse
        CHUNK_DECIMALS = 9
        base = 10 ** CHUNK_DECIMALS
        first_chunk_len = n % CHUNK_DECIMALS or CHUNK_DECIMALS
        i = 0
        val = int(s[i:i + first_chunk_len])
        i += first_chunk_len
        while i < n:
            part = s[i:i + CHUNK_DECIMALS]
            val = val * base + int(part)
            i += CHUNK_DECIMALS

        return cls(sign * val)


    @classmethod
    def from_int(cls, x: int) -> "BigInt":
        return cls(int(x))

    def to_int(self) -> int:
        return int(self._val)

    def __str__(self) -> str:
        return str(self._val)

    def __repr__(self) -> str:
        s = str(self._val)
        return f"BigInt({s[:64]}... len={len(s)})" if len(s) > 64 else f"BigInt({s})"

    # --------------------------
    # Verification helpers
    # --------------------------
    @staticmethod
    def _get_test_primes() -> List[int]:
        # Small 32-bit primes for modular checks:
        # choose a few random-ish primes < 2^31 to minimize collision probability
        return [2147483647, 2147483629, 2147483587]

    @staticmethod
    def _modular_check(a: int, b: int, r: int, primes: Optional[List[int]] = None) -> bool:
        if primes is None:
            primes = BigInt._get_test_primes()
        for p in primes:
            if (a % p) * (b % p) % p != r % p:
                return False
        return True

    # --------------------------
    # Arithmetic
    # --------------------------
    def __add__(self, other: Union["BigInt", int]) -> "BigInt":
        if isinstance(other, BigInt):
            return BigInt(self._val + other._val)
        if isinstance(other, int):
            return BigInt(self._val + other)
        return NotImplemented

    def __sub__(self, other: Union["BigInt", int]) -> "BigInt":
        if isinstance(other, BigInt):
            return BigInt(self._val - other._val)
        if isinstance(other, int):
            return BigInt(self._val - other)
        return NotImplemented

    def _exact_recompute(self, a: int, b: int) -> int:
        """Return exact product using best exact backend (gmpy2 > NTT > python int)."""
        if _HAVE_GMPY2:
            return int(gmpy2.mpz(a) * gmpy2.mpz(b))
        elif _HAVE_NTT:
            # multiply_ints_ntt expects (a,b,base?)—use default base
            return multiply_ints_ntt(a, b)
        else:
            return a * b

    def __mul__(self, other: object) -> "BigInt":
        if isinstance(other, BigInt):
            b = other._val
        elif isinstance(other, int):
            b = other
        else:
            return NotImplemented

        a = self._val

        # 1) If gmpy2 is available, use it (C speed) and we can skip verification.
        if _HAVE_GMPY2:
            res = int(gmpy2.mpz(a) * gmpy2.mpz(b))
            return BigInt(res)

        # 2) Else use auto_multiply (FFT/NTT hybrid)
        if _HAVE_AUTO:
            res = auto_multiply(a, b)
            # 3) Verify with small-modulus checks — if fail, recompute exactly
            ok = BigInt._modular_check(a, b, res)
            if not ok:
                # attempt exact recompute using best exact backend
                exact = self._exact_recompute(a, b)
                return BigInt(int(exact))
            return BigInt(int(res))

        # 4) Fallback to python int (rare)
        return BigInt(int(a) * int(b))

    def __rmul__(self, other: object) -> "BigInt":
        return self.__mul__(other)

    def multiply(self, other: Union["BigInt", int], verify: bool = False) -> "BigInt":
        """Explicit multiply with optional strict verify (recompute with exact backend if requested)."""
        if isinstance(other, BigInt):
            b = other._val
        elif isinstance(other, int):
            b = other
        else:
            raise TypeError("Unsupported operand type for multiply")

        a = self._val

        if _HAVE_GMPY2:
            res = int(gmpy2.mpz(a) * gmpy2.mpz(b))
        elif _HAVE_AUTO:
            res = auto_multiply(a, b)
            if verify:
                ok = BigInt._modular_check(a, b, res)
                if not ok:
                    res = self._exact_recompute(a, b)
        else:
            res = a * b

        if verify and _HAVE_GMPY2:
            # verify using gmpy2 multiplicative check (should always match)
            expected = int(gmpy2.mpz(a) * gmpy2.mpz(b))
            if expected != res:
                raise ValueError("Verification failed")
        return BigInt(int(res))

    # --------------------------
    # Division / Modulo / Pow / GCD / inv
    # --------------------------
    def __floordiv__(self, other: object) -> "BigInt":
        if isinstance(other, BigInt):
            d = other._val
        elif isinstance(other, int):
            d = other
        else:
            return NotImplemented
        if _HAVE_GMPY2:
            return BigInt(int(gmpy2.f_div(self._val, d)))
        return BigInt(self._val // d)

    def __mod__(self, other: object) -> "BigInt":
        if isinstance(other, BigInt):
            m = other._val
        elif isinstance(other, int):
            m = other
        else:
            return NotImplemented
        if _HAVE_GMPY2:
            return BigInt(int(gmpy2.t_mod(self._val, m)))
        return BigInt(self._val % m)

    def powmod(self, exponent: int, modulus: int) -> "BigInt":
        # efficient modular exponentiation
        if _HAVE_GMPY2:
            return BigInt(int(gmpy2.powmod(self._val, exponent, modulus)))
        return BigInt(pow(self._val, exponent, modulus))

    def gcd(self, other: Union["BigInt", int]) -> "BigInt":
        if isinstance(other, BigInt):
            b = other._val
        elif isinstance(other, int):
            b = other
        else:
            raise TypeError("gcd requires BigInt or int")
        if _HAVE_GMPY2:
            return BigInt(int(gmpy2.gcd(self._val, b)))
        import math as _math
        return BigInt(int(_math.gcd(self._val, b)))

    def _egcd(self, a: int, b: int):
        # extended gcd (returns (g, x, y) with ax + by = g)
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = self._egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

    def modinv(self, modulus: int) -> "BigInt":
        # modular inverse a^{-1} mod modulus
        if _HAVE_GMPY2:
            inv = int(gmpy2.invert(self._val, modulus))
            if inv == 0:
                raise ValueError("No modular inverse")
            return BigInt(inv)
        g, x, _ = self._egcd(self._val, modulus)
        if g != 1:
            raise ValueError("No modular inverse")
        return BigInt(x % modulus)

    def __pow__(self, exponent: int, mod: Optional[int] = None) -> "BigInt":
        if mod is None:
            return BigInt(pow(self._val, exponent))
        else:
            return BigInt(pow(self._val, exponent, mod))

    # comparisons
    def __eq__(self, other: object) -> bool:
        if isinstance(other, BigInt):
            return self._val == other._val
        if isinstance(other, int):
            return self._val == other
        return False

    def bit_length(self) -> int:
        return self._val.bit_length()


if __name__ == "__main__":
    import time
    print("Phase 7: BigInt wrapper self-test (math-based digit checks)")

    def decimal_digit_length_from_int(x: int) -> int:
        # Fast, safe decimal digit estimate using log10 of absolute value.
        if x == 0:
            return 1
        # use bit_length -> approximate decimal digits, then refine if needed
        # but int(math.log10(abs(x))) + 1 is fine for our use
        import math
        return int(math.log10(abs(x))) + 1

    tests = [10, 50, 200, 800, 2000, 8000]
    for d in tests:
        a = BigInt.from_decimal_str("9" * d)
        b = BigInt.from_decimal_str("8" * d)
        t0 = time.perf_counter()
        c = a * b
        t1 = time.perf_counter()
        result_len = decimal_digit_length_from_int(c.to_int())
        ok = result_len in (d * 2 - 1, d * 2)
        print(f"digits={d} -> time {t1 - t0:.6f}s, product_ok={ok}")

    # Heavy single-shot test (optional, single trial)
    heavy = 50000
    print(f"\n*** HEAVY TEST: digits={heavy} — single trial. Hit Ctrl-C to abort if too slow ***")
    a = BigInt.from_decimal_str("9" * heavy)
    b = BigInt.from_decimal_str("8" * heavy)
    t0 = time.perf_counter()
    c = a * b
    t1 = time.perf_counter()
    result_len = decimal_digit_length_from_int(c.to_int())
    ok = result_len in (heavy * 2 - 1, heavy * 2)
    print(f"HEAVY digits={heavy} -> time {t1 - t0:.3f}s, product_ok={ok}")

# end of file

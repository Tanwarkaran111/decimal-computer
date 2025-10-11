# decimal_computer/auto_bigint.py
# Connector that chooses the best big-int multiplication strategy and calls the
# appropriate implementation (gmpy2, schoolbook, karatsuba, fft, ntt) available
# in the repository.
from __future__ import annotations
import math
import importlib
import types
from typing import List, Tuple, Optional, Dict, Any

# Try to use gmpy2 if installed
try:
    import gmpy2
    _HAS_GMPY2 = True
except Exception:
    gmpy2 = None
    _HAS_GMPY2 = False

# ----------------------------
# Explicit candidate module paths (edit if your modules live elsewhere)
# ----------------------------
CUSTOM_MODULE_PATHS: Dict[str, List[str]] = {
    "schoolbook": ["decimal_gemm", "phase4.schoolbook", "decimal_gemm.schoolbook", "schoolbook"],
    "karatsuba":  ["phase4.karatsuba", "decimal_gemm.karatsuba", "karatsuba"],
    "fft":        ["phase4.fft_multiply", "decimal_gemm.fft", "fft_multiply"],
    "ntt":        ["phase5.ntt_multiply", "decimal_gemm.ntt", "ntt_multiply"],
}

# Candidate function names we expect from each module (fallback list)
_EXPECTED_FNAMES = ("mul_limbs", "convolve_limbs", "convolve", "mul", "multiply_limbs", "multiply_ints", "multiply_ints_ntt")

def _find_module_and_fn(candidates: List[str]) -> Optional[Tuple[types.ModuleType, str]]:
    """Import first importable module from candidates and choose a likely function name."""
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        # prefer explicit expected names
        for fname in _EXPECTED_FNAMES:
            if hasattr(mod, fname) and callable(getattr(mod, fname)):
                return mod, fname
        # fallback: pick first callable attribute that looks multiply/convolve-like
        for attr in dir(mod):
            low = attr.lower()
            try:
                attr_obj = getattr(mod, attr)
            except Exception:
                continue
            if not callable(attr_obj):
                continue
            if low.startswith("mul") or "conv" in low or "fft" in low or "ntt" in low:
                return mod, attr
    return None

def _find_module_and_fn_prefer(candidates: List[str], preferred_fnames: List[str]) -> Optional[Tuple[types.ModuleType, str]]:
    """Like _find_module_and_fn but prefer an explicit list of function names first."""
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for fname in preferred_fnames:
            if hasattr(mod, fname) and callable(getattr(mod, fname)):
                return mod, fname
        # fallback to generic finder
        for fname in _EXPECTED_FNAMES:
            if hasattr(mod, fname) and callable(getattr(mod, fname)):
                return mod, fname
        for attr in dir(mod):
            low = attr.lower()
            try:
                obj = getattr(mod, attr)
            except Exception:
                continue
            if not callable(obj):
                continue
            if low.startswith("mul") or "conv" in low or "fft" in low or "ntt" in low:
                return mod, attr
    return None

# Detect available modules & functions
_schoolbook = None
_schoolbook_fn = None
_karatsuba = None
_karatsuba_fn = None
_fft = None
_fft_fn = None
_ntt = None
_ntt_fn = None

_detect = _find_module_and_fn_prefer(CUSTOM_MODULE_PATHS.get("schoolbook", []), ["mul_limbs", "int_mul", "int_multiply"])
if _detect:
    _schoolbook, _schoolbook_fn = _detect

_detect = _find_module_and_fn_prefer(CUSTOM_MODULE_PATHS.get("karatsuba", []), ["mul_limbs", "karatsuba_mul", "multiply_ints"])
if _detect:
    _karatsuba, _karatsuba_fn = _detect

_detect = _find_module_and_fn_prefer(CUSTOM_MODULE_PATHS.get("fft", []), ["multiply_ints", "mul_ints", "mul_limbs", "convolve_limbs", "_fft"])
if _detect:
    _fft, _fft_fn = _detect

_detect = _find_module_and_fn_prefer(CUSTOM_MODULE_PATHS.get("ntt", []), ["multiply_ints_ntt", "multiply_ints", "mul_limbs"])
if _detect:
    _ntt, _ntt_fn = _detect

# Limb base. 10**9 is a good compromise: small enough to fit in 64-bit products, big enough to reduce limb count
BASE = 10 ** 9
BASE_DIGITS = 9

# ----------------------------
# Helpers: conversions and normalization
# ----------------------------
def int_to_limbs(n: int, base: int = BASE) -> List[int]:
    """Convert non-negative Python int to limb list (LSB first)."""
    if n == 0:
        return [0]
    if n < 0:
        raise ValueError("int_to_limbs expects non-negative ints")
    limbs: List[int] = []
    b = base
    while n:
        n, rem = divmod(n, b)
        limbs.append(int(rem))
    return limbs

def limbs_to_int(limbs: List[int], base: int = BASE) -> int:
    """Convert limb list (LSB first) back to Python int."""
    b = base
    acc = 0
    for limb in reversed(limbs):
        acc = acc * b + int(limb)
    return acc

def normalize_raw_limbs(raw: List[int], base: int = BASE) -> List[int]:
    """Normalize convolution raw limbs into [0, base) with carries. Returns LSB-first limbs."""
    b = base
    carry = 0
    out: List[int] = []
    for v in raw:
        tot = int(v) + carry
        carry, rem = divmod(tot, b)
        out.append(rem)
    while carry:
        carry, rem = divmod(carry, b)
        out.append(rem)
    # trim leading zeros (but keep at least one digit)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out

def _is_raw_conv_like(raw: List[int], base: int = BASE) -> bool:
    """Heuristic: determine whether output looks like raw conv (may have values >= base or negatives)."""
    for v in raw:
        if v < 0 or v >= base:
            return True
    return False

# ----------------------------
# Strategy selection
# ----------------------------
def choose_strategy_by_digits(digits: int) -> str:
    """Choose strategy name by decimal digit count (very rough heuristics)."""
    if digits < 50:
        return "gmpy2" if _HAS_GMPY2 else "python"
    if digits < 500:
        if _karatsuba:
            return "karatsuba"
        if _schoolbook:
            return "schoolbook"
        return "gmpy2" if _HAS_GMPY2 else "python"
    if digits < 50000:
        if _karatsuba:
            return "karatsuba"
        if _fft:
            return "fft"
        if _ntt:
            return "ntt"
        return "gmpy2" if _HAS_GMPY2 else "python"
    # very large numbers
    if _fft:
        return "fft"
    if _ntt:
        return "ntt"
    if _karatsuba:
        return "karatsuba"
    return "gmpy2" if _HAS_GMPY2 else "python"

# ----------------------------
# Helpers to call modules
# ----------------------------
def _call_module(mod: types.ModuleType, fn_name: str, aL: List[int], bL: List[int], base: int = BASE) -> List[int]:
    """Call a module function assumed to accept limbs. Try common signatures."""
    fn = getattr(mod, fn_name)
    try:
        return fn(aL, bL, base)
    except TypeError:
        try:
            return fn(aL, bL)
        except TypeError:
            # as last resort, call with base again (let exceptions propagate)
            return fn(aL, bL, base)

def _try_call_int_first(mod: types.ModuleType, fn_name: str, a: int, b: int, base: int = BASE) -> Tuple[bool, Any]:
    """
    Try calling mod.fn(a, b) or mod.fn(a, b, base=base).
    Returns (True, result) on success, (False, None) on failure.
    """
    fn = getattr(mod, fn_name)
    try:
        res = fn(a, b)
        return True, res
    except TypeError:
        try:
            res = fn(a, b, base)
            return True, res
        except Exception:
            return False, None
    except Exception:
        # other exceptions propagate (we treat as failure to try limb-based path)
        return False, None

# ----------------------------
# Core API: multiply two Python integers using chosen backend
# ----------------------------
def mul_bigint(a: int, b: int, strategy: str = "auto") -> int:
    """Multiply two Python ints using selected algorithm. strategy='auto' chooses heuristically."""
    if a == 0 or b == 0:
        return 0

    # choose strategy if auto
    if strategy == "auto":
        bits = max(a.bit_length(), b.bit_length())
        dec_digits = max(1, int(math.floor(bits * math.log10(2))) + 1)
        strategy = choose_strategy_by_digits(dec_digits)

    # gmpy2 fast-path
    if strategy == "gmpy2" and _HAS_GMPY2:
        return int(gmpy2.mul(gmpy2.mpz(a), gmpy2.mpz(b)))

    if strategy == "python":
        return a * b

    # Convert sign and absolute values
    sign = 1
    if a < 0:
        a = -a
        sign *= -1
    if b < 0:
        b = -b
        sign *= -1

    a_limbs = int_to_limbs(a, BASE)
    b_limbs = int_to_limbs(b, BASE)

    raw: Optional[List[int]] = None

    # Try explicit strategies with fallbacks
    try:
        if strategy == "schoolbook" and _schoolbook:
            raw = _call_module(_schoolbook, _schoolbook_fn, a_limbs, b_limbs, BASE)

        elif strategy == "karatsuba" and _karatsuba:
            raw = _call_module(_karatsuba, _karatsuba_fn, a_limbs, b_limbs, BASE)

        elif strategy == "fft" and _fft:
            # Prefer integer-level function if module provides one
            ok, res = _try_call_int_first(_fft, _fft_fn, a, b, BASE)
            if ok:
                if isinstance(res, int):
                    return res if sign > 0 else -res
                # if returned limbs or list, treat as raw
                raw = list(res)
            else:
                raw = _call_module(_fft, _fft_fn, a_limbs, b_limbs, BASE)

        elif strategy == "ntt" and _ntt:
            ok, res = _try_call_int_first(_ntt, _ntt_fn, a, b, BASE)
            if ok:
                if isinstance(res, int):
                    return res if sign > 0 else -res
                raw = list(res)
            else:
                raw = _call_module(_ntt, _ntt_fn, a_limbs, b_limbs, BASE)

        else:
            # fallback precedence
            if _fft:
                ok, res = _try_call_int_first(_fft, _fft_fn, a, b, BASE)
                if ok:
                    if isinstance(res, int):
                        return res if sign > 0 else -res
                    raw = list(res)
                else:
                    raw = _call_module(_fft, _fft_fn, a_limbs, b_limbs, BASE)
            elif _ntt:
                ok, res = _try_call_int_first(_ntt, _ntt_fn, a, b, BASE)
                if ok:
                    if isinstance(res, int):
                        return res if sign > 0 else -res
                    raw = list(res)
                else:
                    raw = _call_module(_ntt, _ntt_fn, a_limbs, b_limbs, BASE)
            elif _karatsuba:
                raw = _call_module(_karatsuba, _karatsuba_fn, a_limbs, b_limbs, BASE)
            elif _schoolbook:
                raw = _call_module(_schoolbook, _schoolbook_fn, a_limbs, b_limbs, BASE)
            elif _HAS_GMPY2:
                return int(gmpy2.mul(gmpy2.mpz(a), gmpy2.mpz(b)))
            else:
                return a * b

    except Exception:
        # If an algorithm throws, fallback to gmpy2/python
        if _HAS_GMPY2:
            return int(gmpy2.mul(gmpy2.mpz(a), gmpy2.mpz(b)))
        return a * b

    # If raw is still None, fallback
    if raw is None:
        if _HAS_GMPY2:
            return int(gmpy2.mul(gmpy2.mpz(a), gmpy2.mpz(b)))
        return a * b

    # Normalize / interpret raw result
    if _is_raw_conv_like(raw, BASE):
        normalized = normalize_raw_limbs(raw, BASE)
    else:
        normalized = [int(x) for x in raw]
        while len(normalized) > 1 and normalized[-1] == 0:
            normalized.pop()

    result = limbs_to_int(normalized, BASE)
    return result if sign > 0 else -result

# ----------------------------
# Vector convenience: multiply two lists elementwise (supports batching + gmpy2 preconv)
# ----------------------------
def mul_bigint_list(a_list: List[int], b_list: List[int], strategy: str = "auto", batch: int = 0) -> List[int]:
    """
    Multiply lists of Python ints elementwise.
    - If strategy == 'auto' it resolves to gmpy2 for lists when available.
    - If strategy == 'gmpy2' and gmpy2 available: pre-convert in batches to mpz to limit peak memory.
    - batch=0 => convert entire list at once (fastest). batch>0 => process chunked.
    """
    if len(a_list) != len(b_list):
        raise ValueError("length mismatch")

    if strategy == "auto":
        strategy = "gmpy2" if _HAS_GMPY2 else "python"

    n = len(a_list)

    if strategy == "gmpy2" and _HAS_GMPY2:
        import gmpy2
        if batch is None or batch <= 0:
            a_mp = [gmpy2.mpz(x) for x in a_list]
            b_mp = [gmpy2.mpz(x) for x in b_list]
            return [int(gmpy2.mul(x, y)) for x, y in zip(a_mp, b_mp)]
        else:
            out: List[int] = []
            for i in range(0, n, batch):
                j = min(n, i + batch)
                a_mp = [gmpy2.mpz(x) for x in a_list[i:j]]
                b_mp = [gmpy2.mpz(x) for x in b_list[i:j]]
                out.extend(int(gmpy2.mul(x, y)) for x, y in zip(a_mp, b_mp))
            return out

    # fallback per-element
    out: List[int] = []
    for ai, bi in zip(a_list, b_list):
        out.append(mul_bigint(ai, bi, strategy=strategy))
    return out

# ----------------------------
# Slice helper for orchestrators / workers
# ----------------------------
def mul_bigint_slice(a_seq, b_seq, start: int, end: int, strategy: str = "auto"):
    """Multiply slice start:end using mul_bigint_list. Returns list of ints."""
    suba = a_seq[start:end]
    subb = b_seq[start:end]
    return mul_bigint_list(suba, subb, strategy=strategy)

# ----------------------------
# Availability report
# ----------------------------
def availability_report() -> dict:
    return {
        "gmpy2": _HAS_GMPY2,
        "schoolbook": _schoolbook is not None,
        "schoolbook_fn": _schoolbook_fn,
        "karatsuba": _karatsuba is not None,
        "karatsuba_fn": _karatsuba_fn,
        "fft": _fft is not None,
        "fft_fn": _fft_fn,
        "ntt": _ntt is not None,
        "ntt_fn": _ntt_fn,
        "base": BASE,
        "base_digits": BASE_DIGITS,
    }

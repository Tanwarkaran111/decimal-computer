"""Wrappers to normalize signatures of phase2 GEMM implementations.

Provide a small, robust adapter layer so callers (auto_runtime) can call
any algorithm with a consistent signature:

    func(A, B, cutoff=None, block_size=None, procs=None)

Wrappers attempt to call the underlying implementation using common
keyword names and fall back to positional arguments where reasonable.
If an implementation is missing, the wrapper raises RuntimeError.

Add this file to the repo and update phase3.auto_runtime._ALGOS to
use these wrappers instead of calling phase2 functions directly.
"""
from typing import Any, Callable, Optional
import traceback

# default fallbacks
DEFAULT_CUTOFF = 64
DEFAULT_BLOCK = 64
DEFAULT_PROCS = 4

# try to import backends from phase2; if not available set to None
try:
    from phase2.parallel_karatsuba import karatsuba_gemm as _karatsuba_gemm
except Exception:
    try:
        from phase2.parallel_karatsuba import parallel_multiply as _karatsuba_gemm
    except Exception:
        _karatsuba_gemm = None

try:
    from phase2.block_strassen import block_strassen as _block_strassen
except Exception:
    _block_strassen = None

try:
    # blocked schoolbook used in tuning/benchmarks
    from phase2.parallel_karatsuba import _multiply_block_rows as _schoolbook_blocked
except Exception:
    _schoolbook_blocked = None

try:
    from phase2.decimal_gemm import decimal_gemm_naive as _decimal_gemm_naive
except Exception:
    _decimal_gemm_naive = None


def _try_call(func: Callable, A, B, /, *, cutoff: Optional[int] = None, block_size: Optional[int] = None, procs: Optional[int] = None) -> Any:
    """Attempt to call func using several common kw/positional signatures.

    Order of attempts:
      1. func(A, B)
      2. func(A, B, cutoff)
      3. func(A, B, cutoff=...)
      4. func(A, B, base_cutoff=...)
      5. func(A, B, block_size=...)
      6. func(A, B, b=...)
      7. func(A, B, procs=...)
      8. func(A, B, **kwargs) where kwargs contains some of the keys

    If all attempts raise TypeError due to signature mismatch, the
    original exception is re-raised so the caller can see the cause.
    """
    # quick plain call
    try:
        return func(A, B)
    except TypeError:
        pass

    # positional third arg
    if cutoff is not None:
        try:
            return func(A, B, cutoff)
        except TypeError:
            pass

    # common kw names for cutoffs
    for kw in ("cutoff", "base_cutoff", "base_cutoff_n", "base"):
        if cutoff is not None:
            try:
                return func(A, B, **{kw: cutoff})
            except TypeError:
                pass

    # block size positional/kw
    if block_size is not None:
        try:
            return func(A, B, block_size)
        except TypeError:
            pass
        for kw in ("block_size", "b", "bs", "tile"):
            try:
                return func(A, B, **{kw: block_size})
            except TypeError:
                pass

    # procs / parallelism kw
    if procs is not None:
        for kw in ("procs", "workers", "nprocs", "num_procs"):
            try:
                return func(A, B, **{kw: procs})
            except TypeError:
                pass

    # last resort: attempt calling with all available kwargs
    kwargs = {}
    if cutoff is not None:
        kwargs.update({"cutoff": cutoff})
    if block_size is not None:
        kwargs.update({"block_size": block_size})
    if procs is not None:
        kwargs.update({"procs": procs})
    if kwargs:
        try:
            return func(A, B, **kwargs)
        except TypeError:
            pass

    # If we reached here, re-raise a TypeError to help debugging
    raise TypeError(f"Could not call backend {getattr(func,'__name__',repr(func))} with tried signatures")


def karatsuba_wrapper(A, B, *, cutoff: Optional[int] = None, block_size: Optional[int] = None, procs: Optional[int] = None):
    """Wrapper for karatsuba implementation.

    karatsuba typically ignores block_size; allows procs for parallel
    karatsuba variants.
    """
    if _karatsuba_gemm is None:
        raise RuntimeError("karatsuba backend not available")
    return _try_call(_karatsuba_gemm, A, B, cutoff=cutoff or DEFAULT_CUTOFF, procs=procs or DEFAULT_PROCS)


def strassen_wrapper(A, B, *, cutoff: Optional[int] = None, block_size: Optional[int] = None, procs: Optional[int] = None):
    """Wrapper for Strassen/block_strassen implementation.

    Strassen backends often accept a cutoff parameter to stop recursion.
    """
    if _block_strassen is None:
        raise RuntimeError("strassen backend not available")
    return _try_call(_block_strassen, A, B, cutoff=cutoff or DEFAULT_CUTOFF)


def schoolbook_blocked_wrapper(A, B, *, cutoff: Optional[int] = None, block_size: Optional[int] = None, procs: Optional[int] = None):
    """Wrapper for blocked schoolbook implementation.

    The benchmarks use _multiply_block_rows which accepts a block size
    (positional or kw). We also accept `cutoff` to be flexible.
    """
    if _schoolbook_blocked is None:
        raise RuntimeError("schoolbook_blocked backend not available")
    return _try_call(_schoolbook_blocked, A, B, block_size=block_size or DEFAULT_BLOCK)


def decimal_naive_wrapper(A, B, *, cutoff: Optional[int] = None, block_size: Optional[int] = None, procs: Optional[int] = None):
    """Wrapper for decimal_naive (reference) implementation."""
    if _decimal_gemm_naive is None:
        raise RuntimeError("decimal_naive backend not available")
    return _try_call(_decimal_gemm_naive, A, B)


# convenience mapping to use from auto_runtime
WRAPPERS = {
    "karatsuba": karatsuba_wrapper,
    "strassen": strassen_wrapper,
    "schoolbook_blocked": schoolbook_blocked_wrapper,
    "decimal_naive": decimal_naive_wrapper,
}

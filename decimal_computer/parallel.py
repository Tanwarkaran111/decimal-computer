# decimal_computer/parallel.py
"""
Parallel multiply front-end.

This module exposes `mul_vectors_parallel(a_list, b_list, *, chunksize, max_workers, backend)`
and defaults to a robust, working Python-worker backend that avoids the C-extension
mismatch you experienced. If you later want the older C-based worker backend, you
can switch `backend="ckernel"` explicitly (not recommended until the C worker issues
are fixed).
"""
from __future__ import annotations
from typing import Sequence, Union, Any
from array import array

# Public Scalar type
from .fastdecimal import FastDecimal
Scalar = Union[int, float, str, FastDecimal]

# Prefer the safe Python-worker parallel backend (works reliably)
from .parallel_pyworkers import mul_vectors_parallel_py

# Expose mul_vectors_raw for callers that want the single-process C kernel result
from .vector_ops import mul_vectors_raw

def mul_vectors_parallel(a_list: Sequence[Scalar],
                         b_list: Union[Sequence[Scalar], Scalar],
                         *,
                         chunksize: int = 100_000,
                         max_workers: int | None = None,
                         backend: str = "pyworkers") -> array:
    """
    Parallel elementwise multiply returning array('q') quantized to active context.scale.

    Parameters
    ----------
    a_list, b_list : sequences or scalar broadcast
        Inputs (FastDecimal-compatible).
    chunksize : int
        Per-worker chunk size.
    max_workers : int | None
        Number of workers for process pool (None => default).
    backend : {"pyworkers", "ckernel", "raw"}
        - "pyworkers": use the robust pure-Python worker implementation (default).
        - "ckernel": try the C-kernel process-based parallel implementation (may still be broken).
        - "raw": run the single-process mul_vectors_raw (no parallelism).
    """
    if backend == "raw":
        # Single-process C kernel; no parallelism
        return mul_vectors_raw(a_list, b_list)

    if backend == "pyworkers":
        # Use the reliable multiprocessing Python-worker backend (avoids C kernel in workers)
        return mul_vectors_parallel_py(a_list, b_list, chunksize=chunksize, max_workers=max_workers)

    if backend == "ckernel":
        # Attempt to import the legacy parallel implementation (if present).
        # This path is provided for exploration, but may reproduce the mismatch you saw.
        try:
            from .parallel import mul_vectors_parallel as legacy_mp  # type: ignore
        except Exception as exc:
            raise RuntimeError("C-kernel parallel backend not available: %s" % (exc,))
        return legacy_mp(a_list, b_list, chunksize=chunksize, max_workers=max_workers)

    raise ValueError("Unknown backend: %r" % (backend,))

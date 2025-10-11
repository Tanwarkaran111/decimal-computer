"""
parallel_pyworker_v6.py

Parallel orchestration for elementwise multiplication using the hybrid Cython backend
(decimal_computer.parallel_cworker). This module splits work into chunks, runs worker
processes, and merges results. It includes a smart hybrid orchestrator that automatically
chooses single-process vs parallel execution based on input size and a small sampling
heuristic for large integers.

API
---
- parallel_mul(a, b, n_workers=4, chunk_size=None, return_type='auto')
    Compute elementwise product of a and b in parallel (previous behavior).

- parallel_mul_smart(a, b, n_workers=4, chunk_size=None, return_type='auto', parallel_threshold=200000)
    Smart wrapper that runs single-process fast-hotpath for small inputs and parallel
    for large inputs. It also samples the first few elements to detect very large ints
    (to avoid surprises) and logs which path was chosen via return metadata if requested.

Notes
-----
- This orchestrator delegates numeric work to decimal_computer.parallel_cworker which
  performs hybrid int64 / big-int selection per chunk.
- For Windows (spawn start) large inputs may be copied to workers when the pool is
  created; consider shared_memory or using a long-lived worker process for repeated calls.
"""

from __future__ import annotations
import math
from multiprocessing import Pool
from typing import Any, Iterable, List, Tuple, Optional
import array
import itertools

# Import the compiled hybrid backend
from decimal_computer import parallel_cworker as pc

# Globals used inside worker processes (set by initializer)
_GLOBAL_A = None
_GLOBAL_B = None


def _init_worker(a_obj: Any, b_obj: Any) -> None:
    """Initializer for worker processes: bind global references to input arrays.

    Using an initializer avoids pickling full chunk data for every task — we only
    pickle the two objects once per worker (as initargs). Child tasks will receive
    small tuples of (start, end) which is cheap.
    """
    global _GLOBAL_A, _GLOBAL_B
    _GLOBAL_A = a_obj
    _GLOBAL_B = b_obj


def _worker_apply_range(rng: Tuple[int, int]) -> Any:
    """Worker task executed in each child process.

    `rng` is a (start, end) pair and the worker uses the global _GLOBAL_A/_GLOBAL_B
    plus the compiled backend's mul_int64_slice to compute the partial result.

    Returns whatever the backend returns for that chunk: typically an array.array('q')
    (fast int64 path) or a Python list of ints (big-int path).
    """
    start, end = rng
    # pc.mul_int64_slice will internally choose int64 or big-int path per chunk
    return pc.mul_int64_slice(_GLOBAL_A, _GLOBAL_B, start, end)


def _make_ranges(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Split 0..n into ranges of length chunk_size (last chunk may be shorter)."""
    ranges = []
    for i in range(0, n, chunk_size):
        ranges.append((i, min(n, i + chunk_size)))
    return ranges


def _merge_parts(parts: List[Any], return_type: str = 'auto') -> Any:
    """Merge partial results from workers into a single result.

    - parts: list of chunk outputs. Each chunk output may be array.array('q') or list.
    - return_type: 'auto' => return backend result type per chunk, else 'list' => always list
    """
    if not parts:
        return [] if return_type == 'list' else array.array('q')

    # If user requested a list, just flatten everything into a list
    if return_type == 'list':
        out: List[int] = []
        for p in parts:
            if isinstance(p, array.array):
                out.extend(list(p))
            elif isinstance(p, list):
                out.extend(p)
            else:
                out.extend(list(p))
        return out

    # return_type == 'auto'
    # If all parts are array.array -> create one array and extend
    if all(isinstance(p, array.array) for p in parts):
        out_arr = array.array('q')
        for p in parts:
            out_arr.extend(p)
        return out_arr

    # If all parts are lists -> flatten into list
    if all(isinstance(p, list) for p in parts):
        out: List[int] = []
        for p in parts:
            out.extend(p)
        return out

    # Mixed types: convert everything to list (safe and consistent)
    out = []
    for p in parts:
        if isinstance(p, array.array):
            out.extend(list(p))
        elif isinstance(p, list):
            out.extend(p)
        else:
            out.extend(list(p))
    return out


def parallel_mul(
    a: Iterable[int],
    b: Iterable[int],
    n_workers: int = 4,
    chunk_size: Optional[int] = None,
    return_type: str = 'auto',
) -> Any:
    """Compute elementwise product of a and b in parallel.

    Parameters
    ----------
    a, b : sequences or array-like
        Inputs must have equal length.
    n_workers : int
        Number of parallel worker processes to use.
    chunk_size : Optional[int]
        Number of elements per chunk. If None, computed automatically.
    return_type : str
        'auto' (default) -> return array if fast path used for all chunks else list
        'list' -> always return Python list

    Returns
    -------
    array.array('q') or list
        Merged result (type depends on backend decisions and return_type).
    """
    n = len(a)
    if n != len(b):
        raise ValueError('a and b must have same length')

    if n == 0:
        return [] if return_type == 'list' else array.array('q')

    # Decide chunk size
    if chunk_size is None:
        chunk_size = math.ceil(n / float(max(1, n_workers)))

    ranges = _make_ranges(n, chunk_size)

    with Pool(processes=n_workers, initializer=_init_worker, initargs=(a, b)) as p:
        parts = p.map(_worker_apply_range, ranges)

    return _merge_parts(parts, return_type=return_type)


# Convenience wrapper that always returns a list
def parallel_mul_list(a: Iterable[int], b: Iterable[int], n_workers: int = 4, chunk_size: Optional[int] = None) -> List[int]:
    return parallel_mul(a, b, n_workers=n_workers, chunk_size=chunk_size, return_type='list')


# ---------------------------
# Smart hybrid orchestrator
# ---------------------------

def _sample_contains_bigints(a: Iterable[int], sample_size: int = 10, bit_limit: int = 62) -> bool:
    """Return True if a sample of elements contains values that likely won't fit in signed 64-bit.

    - sample only inspects up to `sample_size` items (cheap). If `a` is not subscriptable,
      it will create an iterator and consume up to sample_size items (not the whole sequence).
    - bit_limit default 62 leaves some headroom before overflow.
    """
    it = iter(a)
    count = 0
    try:
        while count < sample_size:
            v = next(it)
            try:
                iv = int(v)
            except Exception:
                # Non-int-like — treat as big
                return True
            if abs(iv) >= (1 << bit_limit):
                return True
            count += 1
    except StopIteration:
        pass
    return False


def parallel_mul_smart(
    a: Iterable[int],
    b: Iterable[int],
    n_workers: int = 4,
    chunk_size: Optional[int] = None,
    return_type: str = 'auto',
    parallel_threshold: int = 200000,
) -> Any:
    """Smart wrapper that chooses single-process or parallel execution.

    Decision logic:
      1. If len(a) < parallel_threshold -> run single-process pc.mul_int64_full for minimal overhead.
      2. If len(a) >= parallel_threshold -> use parallel_mul (split into chunks)
      3. If sampling detects very large integers in the first few elements, prefer parallel if size
         is large, otherwise still use single-process for small arrays.

    This provides good default behavior on Windows (spawn) where process overhead is non-trivial.
    """
    n = len(a)
    if n != len(b):
        raise ValueError('a and b must have same length')

    # Quick path for trivial sizes
    if n == 0:
        return [] if return_type == 'list' else array.array('q')

    # Sample for big ints (cheap, only samples first few elements)
    has_big = _sample_contains_bigints(a, sample_size=8, bit_limit=62)

    # If below threshold, run single-process for minimal overhead
    if n < parallel_threshold:
        # If inputs are small but contain big ints, still use single-process big-int path
        return pc.mul_int64_full(a, b)

    # Otherwise, use parallel path
    return parallel_mul(a, b, n_workers=n_workers, chunk_size=chunk_size, return_type=return_type)


# Expose the smart function under a convenient name
parallel_mul_auto = parallel_mul_smart

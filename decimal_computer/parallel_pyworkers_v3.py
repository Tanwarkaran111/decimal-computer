from concurrent.futures import ProcessPoolExecutor, as_completed
from array import array
import multiprocessing.shared_memory as shm
import os
import time
from typing import Sequence, List, Tuple, Optional
import math

# --- helpers -----------------------------------------------------------------

def _fits_signed_64bit_sample(ints_sample: Sequence[int], threshold: int = 1024) -> bool:
    """Return True if every value in (a sample of) ints_sample fits signed int64."""
    limit = (1 << 63) - 1
    max_check = min(len(ints_sample), threshold)
    for i in range(max_check):
        v = ints_sample[i]
        if v < -((1 << 63)) or v > limit:
            return False
    return True

def _smart_chunks_indices(n: int, workers: int, base_chunksize_hint: Optional[int] = None) -> List[Tuple[int, int]]:
    """Create chunk indices with front-loaded larger chunks then taper to keep workers busy."""
    if base_chunksize_hint is None or base_chunksize_hint <= 0:
        base = max(1, n // max(1, (workers * 3)))
    else:
        base = base_chunksize_hint
    chunks = []
    done = 0
    while done < n:
        frac = done / n if n else 0.0
        if frac < 0.6:
            size = base * 2
        elif frac < 0.9:
            size = base
        else:
            size = max(1, base // 2)
        end = min(done + size, n)
        chunks.append((done, end))
        done = end
    return chunks

def _worker_mul_shared(start: int, end: int, a_name: str, b_name: str, length: int) -> List[int]:
    a_shm = shm.SharedMemory(name=a_name)
    b_shm = shm.SharedMemory(name=b_name)
    a_arr = array('q')
    b_arr = array('q')
    a_arr.frombytes(a_shm.buf[:8 * length])
    b_arr.frombytes(b_shm.buf[:8 * length])
    out = []
    for i in range(start, end):
        out.append(int(a_arr[i]) * int(b_arr[i]))
    a_shm.close()
    b_shm.close()
    return out

def _worker_mul_fallback(start: int, end: int, a_segment: list, b_segment: list) -> List[int]:
    out = []
    for ai, bi in zip(a_segment, b_segment):
        out.append(int(ai.int_value) * int(bi.int_value))
    return out

# --- main adaptive function --------------------------------------------------

def mul_vectors_parallel_py_v3(a_list: Sequence,
                                b_list: Sequence,
                                *,
                                chunksize: Optional[int] = None,
                                max_workers: Optional[int] = None,
                                prefer_shared: bool = True,
                                sample_size: int = 2048,
                                timeout: Optional[float] = None) -> List[int]:
    """
    Adaptive parallel multiply for FastDecimal-like objects.

    Strategy:
    - Inspect a small sample of integer values to decide whether values fit into signed 64-bit.
    - If they fit, use shared memory arrays of 64-bit signed ints (fast, low-pickle overhead).
    - If not, use a fallback where we send slices to workers, but adapt the number of workers
      down when integers are large (to reduce pickling overhead).
    - Chunking tries to front-load larger chunks then taper to keep workers busy.
    """
    if len(a_list) != len(b_list):
        raise ValueError("a and b must have the same length")
    n = len(a_list)
    if n == 0:
        return []

    cpu = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu

    # Build a small sample of ints without converting the entire lists
    actual_sample = min(sample_size, max(32, n))
    sample_ints = []
    for i in range(actual_sample):
        try:
            sample_ints.append(int(a_list[i].int_value))
            sample_ints.append(int(b_list[i].int_value))
        except Exception:
            # If object doesn't have int_value attribute, fallback to direct int()
            sample_ints.append(int(a_list[i]))
            sample_ints.append(int(b_list[i]))

    # Decide if shared-memory (signed 64) is possible
    can_shared = False
    if prefer_shared and _fits_signed_64bit_sample(sample_ints):
        # quick full-scan check (stop early on overflow)
        try:
            limit = (1 << 63) - 1
            for i in range(n):
                ai = int(a_list[i].int_value) if hasattr(a_list[i], "int_value") else int(a_list[i])
                bi = int(b_list[i].int_value) if hasattr(b_list[i], "int_value") else int(b_list[i])
                if ai < -((1 << 63)) or ai > limit or bi < -((1 << 63)) or bi > limit:
                    raise OverflowError
            can_shared = True
        except OverflowError:
            can_shared = False
        except Exception:
            # any strange issue -> fallback
            can_shared = False

    t0 = time.perf_counter()
    results: List[int] = []

    if can_shared:
        # convert to arrays of 'q' and place in shared memory
        a_arr = array('q', (int(x.int_value) if hasattr(x, "int_value") else int(x) for x in a_list))
        b_arr = array('q', (int(x.int_value) if hasattr(x, "int_value") else int(x) for x in b_list))
        a_shm = shm.SharedMemory(create=True, size=8 * n)
        b_shm = shm.SharedMemory(create=True, size=8 * n)
        try:
            a_shm.buf[:8 * n] = a_arr.tobytes()
            b_shm.buf[:8 * n] = b_arr.tobytes()
            if chunksize is None:
                chunksize = max(1, n // (max_workers * 3))
            idxs = _smart_chunks_indices(n, max_workers, base_chunksize_hint=chunksize)
            fut_map = {}
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for i, (s, e) in enumerate(idxs):
                    fut = exe.submit(_worker_mul_shared, s, e, a_shm.name, b_shm.name, n)
                    fut_map[fut] = i
                segs = [None] * len(idxs)
                for fut in as_completed(fut_map, timeout=timeout):
                    segs[fut_map[fut]] = fut.result()
            for seg in segs:
                results.extend(seg)
        finally:
            a_shm.close()
            a_shm.unlink()
            b_shm.close()
            b_shm.unlink()
    else:
        # Fallback: big ints => reduce workers to reduce pickle overhead
        avg_bitlen = 0
        cnt = len(sample_ints) or 1
        for v in sample_ints:
            avg_bitlen += (v.bit_length() if isinstance(v, int) else int(v).bit_length())
        avg_bitlen = avg_bitlen / cnt

        if avg_bitlen > 16384:
            target_workers = max(1, cpu // 4)
        elif avg_bitlen > 4096:
            target_workers = max(1, cpu // 2)
        elif avg_bitlen > 1024:
            target_workers = max(1, cpu * 3 // 4)
        else:
            target_workers = max_workers

        target_workers = int(max(1, min(target_workers, max_workers)))

        if chunksize is None:
            chunksize = max(1, n // (target_workers * 2))

        idxs = _smart_chunks_indices(n, target_workers, base_chunksize_hint=chunksize)
        fut_map = {}
        segs = [None] * len(idxs)
        # defensive: if any worker raises OverflowError while pickling/processing, reduce workers
        try:
            with ProcessPoolExecutor(max_workers=target_workers) as exe:
                for i, (s, e) in enumerate(idxs):
                    # send slices (less pickling than whole lists)
                    fut = exe.submit(_worker_mul_fallback, s, e, a_list[s:e], b_list[s:e])
                    fut_map[fut] = i
                for fut in as_completed(fut_map, timeout=timeout):
                    segs[fut_map[fut]] = fut.result()
        except OverflowError:
            # last resort serial fallback to guarantee correctness
            segs = []
            for s, e in idxs:
                segs.append(_worker_mul_fallback(s, e, a_list[s:e], b_list[s:e]))
        for seg in segs:
            results.extend(seg)

    total = round(time.perf_counter() - t0, 3)
    print(f"[main] submitting {len(idxs)} chunk(s) using up to {max_workers} worker(s) on {cpu} CPU(s) (chunksize≈{chunksize})")
    print(f"[main] parallel multiply completed in {total}s")
    return results

# Backwards compatibility: many callers expect mul_vectors_parallel_py name
if 'mul_vectors_parallel_py' not in globals():
    mul_vectors_parallel_py = mul_vectors_parallel_py_v3

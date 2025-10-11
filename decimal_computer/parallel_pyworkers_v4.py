from concurrent.futures import ProcessPoolExecutor, as_completed
from array import array
import multiprocessing.shared_memory as shm
import os
import time
from typing import Sequence, List, Tuple, Optional
import math


def _worker_mul_shared(start: int, end: int, a_name: str, b_name: str, length: int) -> list:
    # Attach to existing shared memory and compute products for slice [start:end)
    a_shm = shm.SharedMemory(name=a_name)
    b_shm = shm.SharedMemory(name=b_name)
    try:
        a_arr = array('q')
        b_arr = array('q')
        a_arr.frombytes(a_shm.buf[:8 * length])
        b_arr.frombytes(b_shm.buf[:8 * length])
        out = [int(a_arr[i]) * int(b_arr[i]) for i in range(start, end)]
    finally:
        a_shm.close()
        b_shm.close()
    return out


def _worker_mul_fallback(start: int, end: int, a_segment: list, b_segment: list) -> list:
    # Fallback for arbitrarily large integers / non-int inputs (pickled to worker)
    out = []
    for ai, bi in zip(a_segment, b_segment):
        # ai/bi are expected to be FastDecimal-like objects with .int_value attribute,
        # but we tolerate plain ints as well.
        try:
            aval = int(ai.int_value)
        except Exception:
            aval = int(ai)
        try:
            bval = int(bi.int_value)
        except Exception:
            bval = int(bi)
        out.append(aval * bval)
    return out


def _smart_chunks_indices(n: int, workers: int, base_chunksize_hint: Optional[int] = None) -> List[Tuple[int, int]]:
    # Create a list of (start, end) indices that front-loads larger chunks then tapers
    if base_chunksize_hint is None or base_chunksize_hint <= 0:
        base = max(1, n // (workers * 3))
    else:
        base = base_chunksize_hint
    chunks = []
    done = 0
    while done < n:
        frac = done / n if n else 0
        if frac < 0.5:
            size = base * 2
        elif frac < 0.85:
            size = base
        else:
            size = max(1, base // 2)
        end = min(done + size, n)
        chunks.append((done, end))
        done = end
    return chunks


def _fits_signed_64bit_max(ints_sample: Sequence[int], threshold: int = 1000) -> bool:
    # check sample values for int64 fit
    max_check = min(len(ints_sample), threshold)
    limit = (1 << 63) - 1
    for i in range(max_check):
        v = ints_sample[i]
        if v < -limit - 1 or v > limit:
            return False
    return True


def mul_vectors_parallel_py(
    a_list: Sequence,
    b_list: Sequence,
    *,
    chunksize: Optional[int] = None,
    max_workers: Optional[int] = None,
    use_shared_when_possible: bool = True,
    timeout: Optional[float] = None,
) -> List[int]:
    """Multiply two sequences elementwise in parallel and return list of ints.

    This implementation will attempt to use shared memory when the integer values
    fit in signed 64-bit to avoid pickling overhead. If values are larger, it
    falls back to sending slices to workers.
    """
    if len(a_list) != len(b_list):
        raise ValueError("a and b must have the same length")
    n = len(a_list)
    if n == 0:
        return []

    cpu = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu

    # Build a small sample of integer representations without converting everything
    sample_count = min(1024, n)
    sample_ints = []
    for i in range(sample_count):
        try:
            sample_ints.append(int(a_list[i].int_value))
        except Exception:
            try:
                sample_ints.append(int(a_list[i]))
            except Exception:
                sample_ints.append(0)
        try:
            sample_ints.append(int(b_list[i].int_value))
        except Exception:
            try:
                sample_ints.append(int(b_list[i]))
            except Exception:
                sample_ints.append(0)

    can_shared = False
    if use_shared_when_possible and _fits_signed_64bit_max(sample_ints):
        # quick verify all (may raise) before allocating huge shared memory
        try:
            for i in range(n):
                v = int(getattr(a_list[i], 'int_value', a_list[i]))
                if v < -(1 << 63) or v > (1 << 63) - 1:
                    raise OverflowError
                v2 = int(getattr(b_list[i], 'int_value', b_list[i]))
                if v2 < -(1 << 63) or v2 > (1 << 63) - 1:
                    raise OverflowError
            can_shared = True
        except Exception:
            can_shared = False

    t0 = time.perf_counter()
    results: List[int] = []

    if can_shared:
        # Build arrays of signed 64-bit integers and place in shared memory
        a_arr = array('q', (int(getattr(x, 'int_value', x)) for x in a_list))
        b_arr = array('q', (int(getattr(x, 'int_value', x)) for x in b_list))
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
            a_shm.close(); a_shm.unlink()
            b_shm.close(); b_shm.unlink()
    else:
        # fallback: avoid over-parallelizing if big ints (pickle heavy)
        avg_bitlen = 0
        for v in sample_ints:
            avg_bitlen += v.bit_length()
        avg_bitlen = avg_bitlen / (len(sample_ints) or 1)
        if avg_bitlen > 4096:
            target_workers = max(1, cpu // 2)
        elif avg_bitlen > 1024:
            target_workers = max(1, cpu * 3 // 4)
        else:
            target_workers = max_workers
        target_workers = int(min(target_workers, max_workers))

        if chunksize is None:
            chunksize = max(1, n // (target_workers * 2))

        idxs = _smart_chunks_indices(n, target_workers, base_chunksize_hint=chunksize)
        fut_map = {}
        with ProcessPoolExecutor(max_workers=target_workers) as exe:
            for i, (s, e) in enumerate(idxs):
                fut = exe.submit(_worker_mul_fallback, s, e, a_list[s:e], b_list[s:e])
                fut_map[fut] = i
            segs = [None] * len(idxs)
            for fut in as_completed(fut_map, timeout=timeout):
                segs[fut_map[fut]] = fut.result()
        for seg in segs:
            results.extend(seg)

    total = round(time.perf_counter() - t0, 3)
    print(f"[main] submitting {len(idxs)} chunk(s) using up to {max_workers} worker(s) on {cpu} CPU(s) (chunksize≈{chunksize})")
    print(f"[main] parallel multiply completed in {total}s")
    return results

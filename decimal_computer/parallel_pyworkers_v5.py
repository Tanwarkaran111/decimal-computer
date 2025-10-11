# decimal_computer/parallel_pyworkers_v5.py
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from array import array
import multiprocessing.shared_memory as shm
import os
import time
from typing import Sequence, List, Tuple, Optional
import math
import gc

# local worker function import (the worker function is pickled by value)
from .parallel_worker import _worker_mul_shared_indices

def _smart_chunks_indices(n: int, workers: int, base_chunksize_hint: Optional[int] = None) -> List[Tuple[int, int]]:
    if base_chunksize_hint is None or base_chunksize_hint <= 0:
        base = max(1, n // (max(1, workers) * 3))
    else:
        base = base_chunksize_hint
    chunks = []
    done = 0
    # front-load larger chunks then taper
    while done < n:
        frac = done / n if n else 0
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

def _fits_signed_64bit_max(ints_sample: Sequence[int], threshold: int = 1000) -> bool:
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
    timeout: Optional[float] = None,
) -> List[int]:
    """
    Parallel multiply using explicit per-task shared-memory slice copying.
    Each worker opens the shared memory by name, copies its slice bytes only,
    and closes the SharedMemory object promptly — this avoids long-lived exported
    buffers and the `BufferError` seen previously.

    Falls back to pickled-slice workers if ints don't fit into 64-bit.
    """
    if len(a_list) != len(b_list):
        raise ValueError("a and b must have the same length")
    n = len(a_list)
    if n == 0:
        return []

    cpu = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu

    # sample ints for shape detection
    sample_count = min(1024, n)
    sample_ints: List[int] = []
    for i in range(sample_count):
        sample_ints.append(int(a_list[i].int_value))
        sample_ints.append(int(b_list[i].int_value))

    # quick check: sample fit into signed 64-bit?
    can_shared = False
    if _fits_signed_64bit_max(sample_ints):
        try:
            for i in range(n):
                v = int(a_list[i].int_value)
                if v < -(1 << 63) or v > (1 << 63) - 1:
                    raise OverflowError
                v2 = int(b_list[i].int_value)
                if v2 < -(1 << 63) or v2 > (1 << 63) - 1:
                    raise OverflowError
            can_shared = True
        except OverflowError:
            can_shared = False

    t0 = time.perf_counter()
    results: List[int] = []
    idxs: List[Tuple[int, int]] = []

    if can_shared:
        # create arrays and write bytes to shared memory
        a_arr = array("q", (int(x.int_value) for x in a_list))
        b_arr = array("q", (int(x.int_value) for x in b_list))

        # allocate shared memory and copy bytes
        a_shm = shm.SharedMemory(create=True, size=8 * n)
        b_shm = shm.SharedMemory(create=True, size=8 * n)

        # copy bytes via an intermediate bytes object to avoid exported memoryviews
        tmp = a_arr.tobytes()
        a_shm.buf[: 8 * n] = tmp
        del tmp
        tmp = b_arr.tobytes()
        b_shm.buf[: 8 * n] = tmp
        del tmp
        gc.collect()

        # choose chunking
        if chunksize is None:
            chunksize = max(1, n // (max_workers * 3))
        idxs = _smart_chunks_indices(n, max_workers, base_chunksize_hint=chunksize)

        try:
            # Submit tasks: each task will open shared memory by name, copy its slice, then close it.
            fut_map = {}
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for i, (s, e) in enumerate(idxs):
                    fut = exe.submit(_worker_mul_shared_indices, s, e, a_shm.name, b_shm.name, n)
                    fut_map[fut] = i

                segs = [None] * len(idxs)
                for fut in as_completed(fut_map, timeout=timeout):
                    segs[fut_map[fut]] = fut.result()

            # collect segments
            for seg in segs:
                results.extend(seg)
        finally:
            # drop parent arrays, force GC, then safe close/unlink
            try:
                del a_arr
                del b_arr
            except Exception:
                pass
            gc.collect()
            # small sleep to allow worker processes finalization if needed
            time.sleep(0.01)

            def _safe_close_unlink(shm_obj):
                if shm_obj is None:
                    return
                # attempt close with retries
                for attempt in range(6):
                    try:
                        shm_obj.close()
                        break
                    except BufferError:
                        gc.collect()
                        time.sleep(0.02 * (attempt + 1))
                    except Exception:
                        break
                # attempt unlink with retries
                for attempt in range(6):
                    try:
                        shm_obj.unlink()
                        break
                    except BufferError:
                        gc.collect()
                        time.sleep(0.02 * (attempt + 1))
                    except FileNotFoundError:
                        break
                    except Exception:
                        break

            _safe_close_unlink(a_shm)
            _safe_close_unlink(b_shm)

    else:
        # fallback: pickled-slice workers (no shared memory)
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
        from .parallel_worker import _worker_mul_shared_indices as _worker_local  # fallback local function not using shared memory
        with ProcessPoolExecutor(max_workers=target_workers) as exe:
            for i, (s, e) in enumerate(idxs):
                # fallback: pack slices of objects (slower but safe)
                fut = exe.submit(lambda ss, ee, a_slice, b_slice: [int(ai.int_value) * int(bi.int_value) for ai, bi in zip(a_slice, b_slice)],
                                 s, e, a_list[s:e], b_list[s:e])
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

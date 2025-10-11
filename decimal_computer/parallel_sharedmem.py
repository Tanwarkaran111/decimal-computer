from __future__ import annotations
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Union
from array import array
from .vector_ops import _is_sequence, ensure_fastdec
from .context import get_context
from .rounding import quantize_int
import os

Scalar = Union[int, float, str, "FastDecimal"]

def _worker_sharedmem_write(start: int,
                             a_ints: list,
                             b_ints: list,
                             a_scales: list,
                             b_scales: list,
                             target_scale: int,
                             rounding: str,
                             shm_name: str) -> int:
    from multiprocessing import shared_memory as _shm
    shm = _shm.SharedMemory(name=shm_name)
    buf = shm.buf
    for i, (ai, bi, asc, bsc) in enumerate(zip(a_ints, b_ints, a_scales, b_scales)):
        prod = int(ai) * int(bi)
        orig_scale = int(asc) + int(bsc)
        q = quantize_int(prod, orig_scale=orig_scale, target_scale=target_scale, rounding=rounding)
        idx = start + i
        off = idx * 8
        v = int(q)
        if v < -2**63 or v >= 2**63:
            shm.close()
            raise OverflowError(f"quantized result at pos {idx} does not fit in int64: {v}")
        buf[off:off+8] = v.to_bytes(8, byteorder='little', signed=True)
    shm.close()
    return start

def mul_vectors_sharedmem(a_list: Sequence[Scalar],
                           b_list: Union[Sequence[Scalar], Scalar],
                           *,
                           chunksize: int = 100_000,
                           max_workers: int | None = None) -> array:
    if not _is_sequence(b_list):
        b_seq = [b_list] * len(a_list)
    else:
        b_seq = list(b_list)
        if len(b_seq) != len(a_list):
            raise ValueError("a_list and b_list must have same length")

    n = len(a_list)
    if n == 0:
        return array('q')

    a_ints_all = []
    b_ints_all = []
    a_scales_all = []
    b_scales_all = []
    for ai, bi in zip(a_list, b_seq):
        a_fd = ensure_fastdec(ai)
        b_fd = ensure_fastdec(bi)
        a_ints_all.append(int(a_fd.int_value))
        b_ints_all.append(int(b_fd.int_value))
        a_scales_all.append(int(a_fd.scale))
        b_scales_all.append(int(b_fd.scale))

    ctx = get_context()
    target_scale = ctx.scale
    rounding = ctx.rounding

    total_bytes = n * 8
    shm = shared_memory.SharedMemory(create=True, size=total_bytes)
    shm.buf[:] = b"\x00" * total_bytes

    chunks = [(i, min(i + chunksize, n)) for i in range(0, n, chunksize)]

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for start, stop in chunks:
            a_chunk = a_ints_all[start:stop]
            b_chunk = b_ints_all[start:stop]
            asc = a_scales_all[start:stop]
            bsc = b_scales_all[start:stop]
            fut = exe.submit(_worker_sharedmem_write, start, a_chunk, b_chunk, asc, bsc, target_scale, rounding, shm.name)
            futures[fut] = start

        for fut in as_completed(futures):
            _ = fut.result()

    out = array('q')
    buf = shm.buf
    for i in range(n):
        off = i * 8
        v = int.from_bytes(buf[off:off+8], byteorder='little', signed=True)
        out.append(v)

    shm.close()
    shm.unlink()
    return out

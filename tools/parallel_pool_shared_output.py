from __future__ import annotations
import time
from array import array
from typing import Sequence, Union, List, Tuple
from multiprocessing import Pool, Array, cpu_count
from decimal_computer.vector_ops import _is_sequence, ensure_fastdec
from decimal_computer.context import get_context
from decimal_computer.rounding import quantize_int
from decimal_computer.fastdecimal import FastDecimal

Scalar = Union[int, float, str, FastDecimal]

_A_SHM = None
_B_SHM = None
_A_SC_SHM = None
_B_SC_SHM = None
_OUT_SHM = None
_TARGET_SCALE = None
_ROUNDING = None

def _worker_init(a_shm, b_shm, a_sc_shm, b_sc_shm, out_shm, target_scale, rounding):
    global _A_SHM, _B_SHM, _A_SC_SHM, _B_SC_SHM, _OUT_SHM, _TARGET_SCALE, _ROUNDING
    _A_SHM = a_shm
    _B_SHM = b_shm
    _A_SC_SHM = a_sc_shm
    _B_SC_SHM = b_sc_shm
    _OUT_SHM = out_shm
    _TARGET_SCALE = target_scale
    _ROUNDING = rounding

def _worker_write_range(args: Tuple[int,int]) -> None:
    start, stop = args
    a = _A_SHM
    b = _B_SHM
    a_sc = _A_SC_SHM
    b_sc = _B_SC_SHM
    out = _OUT_SHM
    for i in range(start, stop):
        prod = int(a[i]) * int(b[i])
        orig_scale = int(a_sc[i]) + int(b_sc[i])
        q = quantize_int(prod, orig_scale=orig_scale, target_scale=_TARGET_SCALE, rounding=_ROUNDING)
        out[i] = int(q)
    return None

def mul_vectors_pool_shared_output(a_list: Sequence[Scalar],
                                   b_list: Union[Sequence[Scalar], Scalar],
                                   *,
                                   chunksize: int | None = None,
                                   max_workers: int | None = None,
                                   use_pool: bool = True) -> array:
    if not _is_sequence(b_list):
        b_seq = [b_list] * len(a_list)
    else:
        b_seq = list(b_list)
        if len(b_seq) != len(a_list):
            raise ValueError("a_list and b_list must have same length")

    n = len(a_list)
    if n == 0:
        return array("q")

    a_ints = [int(ensure_fastdec(x).int_value) for x in a_list]
    b_ints = [int(ensure_fastdec(x).int_value) for x in b_seq]
    a_scales = [int(ensure_fastdec(x).scale) for x in a_list]
    b_scales = [int(ensure_fastdec(x).scale) for x in b_seq]

    ctx = get_context()
    tgt_scale = ctx.scale
    rounding = ctx.rounding

    if chunksize is None:
        wc = max_workers or (cpu_count() or 1)
        chunksize = max(1, n // max(1, wc * 4))

    chunks = [(i, min(i + chunksize, n)) for i in range(0, n, chunksize)]

    if not use_pool or (max_workers == 1):
        out = array("q")
        for start, stop in chunks:
            for i in range(start, stop):
                prod = a_ints[i] * b_ints[i]
                orig_scale = a_scales[i] + b_scales[i]
                out.append(int(quantize_int(prod, orig_scale=orig_scale, target_scale=tgt_scale, rounding=rounding)))
        return out

    a_shm = Array('q', a_ints, lock=False)
    b_shm = Array('q', b_ints, lock=False)
    a_sc_shm = Array('i', a_scales, lock=False)
    b_sc_shm = Array('i', b_scales, lock=False)
    out_shm = Array('q', n, lock=False)

    if max_workers is None:
        max_workers = cpu_count() or 1

    pool = Pool(processes=max_workers,
                initializer=_worker_init,
                initargs=(a_shm, b_shm, a_sc_shm, b_sc_shm, out_shm, tgt_scale, rounding))
    try:
        pool.map(_worker_write_range, chunks)
    finally:
        pool.close()
        pool.join()

    out = array("q", out_shm[:])
    return out

def _bench():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    args = parser.parse_args()
    N = args.n
    a = [FastDecimal.from_str('1.23')] * N
    b = [FastDecimal.from_str('4.56')] * N

    print("single baseline...")
    t0 = time.perf_counter()
    mul_vectors_pool_shared_output(a, b, chunksize=args.chunksize, max_workers=1, use_pool=False)
    s1 = time.perf_counter() - t0
    print("single elapsed", round(s1, 3))

    print("pool shared-output...")
    t0 = time.perf_counter()
    mul_vectors_pool_shared_output(a, b, chunksize=args.chunksize, max_workers=args.workers, use_pool=True)
    s2 = time.perf_counter() - t0
    print("pool elapsed", round(s2, 3), "workers", args.workers or cpu_count())

if __name__ == "__main__":
    _bench()

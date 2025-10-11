from __future__ import annotations
import os
import time
from typing import List, Sequence, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from array import array
from decimal_computer.fastdecimal import FastDecimal
from decimal_computer.vector_ops import ensure_fastdec, _is_sequence
from decimal_computer.rounding import quantize_int
from decimal_computer.context import get_context

Scalar = Union[int, float, str, FastDecimal]


def _worker_mul_chunk(a_chunk: List[int],
                      b_chunk: List[int],
                      a_scales: List[int],
                      b_scales: List[int],
                      target_scale: int,
                      rounding: str) -> array:
    from array import array as _array
    from decimal_computer.rounding import quantize_int as _qint

    n = len(a_chunk)
    out = _array("q")
    for i in range(n):
        prod = a_chunk[i] * b_chunk[i]
        orig_scale = a_scales[i] + b_scales[i]
        out.append(_qint(prod, orig_scale, target_scale, rounding))
    return out


def mul_vectors_pool(a_list: Sequence[Scalar],
                     b_list: Union[Sequence[Scalar], Scalar],
                     *,
                     chunksize: int = 100_000,
                     max_workers: int | None = None) -> array:
    if not _is_sequence(b_list):
        b_seq = [b_list] * len(a_list)
    else:
        b_seq = list(b_list)
        if len(a_list) != len(b_seq):
            raise ValueError("a_list and b_list must have same length")

    n = len(a_list)
    if n == 0:
        return array("q")

    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu_count

    ctx = get_context()
    target_scale = ctx.scale
    rounding = ctx.rounding

    a_ints = [int(ensure_fastdec(x).int_value) for x in a_list]
    b_ints = [int(ensure_fastdec(x).int_value) for x in b_seq]
    a_scales = [int(ensure_fastdec(x).scale) for x in a_list]
    b_scales = [int(ensure_fastdec(x).scale) for x in b_seq]

    chunks: List[Tuple[int, int]] = []
    for start in range(0, n, chunksize):
        stop = min(start + chunksize, n)
        chunks.append((start, stop))

    start_time = time.perf_counter()
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for start, stop in chunks:
            fut = exe.submit(_worker_mul_chunk,
                             a_ints[start:stop],
                             b_ints[start:stop],
                             a_scales[start:stop],
                             b_scales[start:stop],
                             target_scale,
                             rounding)
            futures[fut] = start

        for fut in as_completed(futures):
            start = futures[fut]
            results[start] = fut.result()

    out = array("q")
    for start in sorted(results.keys()):
        out.extend(results[start])

    elapsed = time.perf_counter() - start_time
    print(f"[main] pool multiply completed in {elapsed:.3f}s")
    return out


def main(n: int = 200_000):
    from decimal_computer.fastdecimal import FastDecimal

    a = [FastDecimal.from_str("1.23")] * n
    b = [FastDecimal.from_str("4.56")] * n

    print("running single-worker baseline...")
    t0 = time.perf_counter()
    mul_vectors_pool(a, b, chunksize=n, max_workers=1)
    print("single elapsed", round(time.perf_counter() - t0, 3))

    print("running pool (workers=12)...")
    t1 = time.perf_counter()
    mul_vectors_pool(a, b, chunksize=max(1, n // (os.cpu_count() or 12)), max_workers=os.cpu_count())
    print("pool elapsed", round(time.perf_counter() - t1, 3), "workers", os.cpu_count())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    args = p.parse_args()
    main(args.n)

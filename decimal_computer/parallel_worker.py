# decimal_computer/parallel_worker.py
from array import array
import multiprocessing.shared_memory as shm
from typing import List, Sequence, Optional

def _worker_mul_shared_indices(start: int, end: int, a_name: str, b_name: str, length: int):
    """
    Worker task: open shared memory blocks by name, copy only the byte-range for
    [start:end) (so we avoid long-lived exported pointers), convert to array('q'),
    run fast Cython routine if present, otherwise fallback to Python multiply.

    Returns: list or array('q') of products for indices [start:end)
    """
    # open shared memory objects locally (per-task)
    a_shm = shm.SharedMemory(name=a_name)
    b_shm = shm.SharedMemory(name=b_name)
    try:
        # compute byte offsets for the slice
        s_byte = start * 8
        e_byte = end * 8
        if e_byte > len(a_shm.buf) or e_byte > len(b_shm.buf):
            raise ValueError("slice out of range of shared memory block")

        # copy only the slice bytes -> this yields a plain bytes object (no exported memoryview)
        a_bytes = bytes(a_shm.buf[s_byte:e_byte])
        b_bytes = bytes(b_shm.buf[s_byte:e_byte])

        # construct arrays from bytes (no further exported memoryviews)
        a_arr = array('q')
        b_arr = array('q')
        a_arr.frombytes(a_bytes)
        b_arr.frombytes(b_bytes)

        # try to use Cython accelerator if available
        try:
            from decimal_computer import parallel_cworker as _pc  # local import in worker
            # prefer a function that multiplies two arrays of equal length
            if hasattr(_pc, "mul_int64_arrays"):
                # _pc.mul_int64_arrays expects array-like; pass our small arrays
                try:
                    return _pc.mul_int64_arrays(a_arr, b_arr)
                except Exception:
                    # fall through to pure-Python fallback
                    pass
        except Exception:
            # compiled extension not present or import failed -> fallback
            pass

        # Pure-Python fallback: multiply elementwise
        out: List[int] = []
        for i in range(end - start):
            out.append(int(a_arr[i]) * int(b_arr[i]))
        return out

    finally:
        # close the SharedMemory handles on this worker process
        try:
            a_shm.close()
        except Exception:
            pass
        try:
            b_shm.close()
        except Exception:
            pass

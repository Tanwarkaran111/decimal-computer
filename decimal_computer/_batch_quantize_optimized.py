from . import batch_quantize
import time


def batch_quantize_optimized(a_ints, b_ints, a_scales, b_scales, target_scale, rounding):
    """
    Optimized batch quantization dispatcher.
    Tries to use Cython-accelerated batch_quantize_v2 if available,
    otherwise falls back to the pure Python batch_quantize.batch_quantize.
    """
    try:
        from . import batch_quantize_v2
        func = getattr(batch_quantize_v2, "batch_quantize_v2", None)
        if callable(func):
            return func(a_ints, b_ints, a_scales, b_scales, target_scale, rounding)
        else:
            raise ImportError("batch_quantize_v2 not callable")
    except Exception:
        return batch_quantize.batch_quantize(a_ints, b_ints, a_scales, b_scales, target_scale, rounding)


def _bench(n=200000):
    """Quick microbenchmark."""
    from decimal_computer.fastdecimal import FastDecimal
    import random

    a = [FastDecimal.from_str(str(random.uniform(1, 10))) for _ in range(n)]
    b = [FastDecimal.from_str(str(random.uniform(1, 10))) for _ in range(n)]
    ai = [int(x.int_value) for x in a]
    bi = [int(x.int_value) for x in b]
    asc = [int(x.scale) for x in a]
    bsc = [int(x.scale) for x in b]
    t0 = time.perf_counter()
    out = batch_quantize_optimized(ai, bi, asc, bsc, 2, "ROUND_HALF_UP")
    print("len", len(out), "elapsed", round(time.perf_counter() - t0, 3))


# Ensure _batch_quantize_v2 is callable even if compiled symbol is missing.
try:
    if not (_batch_quantize_v2 and callable(_batch_quantize_v2)):
        _batch_quantize_v2 = batch_quantize_optimized
except NameError:
    _batch_quantize_v2 = batch_quantize_optimized


if __name__ == "__main__":
    _bench()

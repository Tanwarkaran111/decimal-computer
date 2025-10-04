# numba_decimal.py
"""
Numba-optimized decimal GEMM prototype.

Usage (example):
    python numba_decimal.py --n 512 --digits 8 --chunk 2 --tile 64
This will run a decimal GEMM on random integer matrices of size n x n,
representing each element as decimal digits (chunked) and performing
a tiled GEMM in Numba.
Results are saved to numba_decimal_results.csv
"""

import argparse
import time
import numpy as np
import csv
from pathlib import Path

try:
    from numba import njit, prange
except Exception as e:
    raise SystemExit("Please install numba (pip install numba) to run this script.") from e


# ---- helpers to split integers into decimal chunks ----
def int_to_chunks_matrix(A_int: np.ndarray, chunk_base: int, n_chunks: int):
    """
    Convert integer matrix A_int (dtype=int64) to chunked decimal representation
    Returns array shape (n, n, n_chunks) dtype=int64, little-endian chunks (least-significant first).
    """
    n = A_int.shape[0]
    C = np.zeros((n, n, n_chunks), dtype=np.int64)
    temp = A_int.copy()
    for k in range(n_chunks):
        C[:, :, k] = temp % chunk_base
        temp //= chunk_base
    return C


def chunks_to_int_matrix(A_chunks: np.ndarray, chunk_base: int):
    n, _, n_chunks = A_chunks.shape
    out = np.zeros((n, n), dtype=np.int64)
    mult = 1
    for k in range(n_chunks):
        out += A_chunks[:, :, k] * mult
        mult *= chunk_base
    return out


# ---- Numba-compiled per-chunk multiply-add (schoolbook on chunks) ----
# Note: numba prefers prange(n) with simple ranges. Using prange with step
# or complex Python objects causes "Failed in nopython mode pipeline (convert to parfors)".
#
# This function parallelizes over rows (i) and performs the full convolution.
# tile parameter is left in the API for compatibility but not used in the kernel
# because earlier tiled prange was causing numba rewrite issues.
@njit(parallel=True, fastmath=True)
def decimal_gemm_numba(A_chunks, B_chunks, chunk_base, tile):
    """
    A_chunks, B_chunks: shape (n, n, n_chunks) int64
    Returns C_chunks of shape (n, n, n_chunks) representing integer result
    This is a simple schoolbook per-chunk convolution with carry handling at the end.
    """
    n = A_chunks.shape[0]
    n_chunks = A_chunks.shape[2]

    # Accumulate convolution results in larger temporary per-chunk positions.
    # We allocate size (n, n, 2*n_chunks) to avoid index overflow when adding u+v.
    Acc = np.zeros((n, n, 2 * n_chunks), dtype=np.int64)

    # Parallel outer loop over rows. Inner loops are standard Python-range loops
    # which numba compiles well into nopython.
    for i in prange(n):
        for k in range(n):
            # get reference views to speed up indexing
            arow = A_chunks[i, k]  # shape (n_chunks,)
            for j in range(n):
                brow = B_chunks[k, j]  # shape (n_chunks,)
                # simple convolution accumulation
                for u in range(n_chunks):
                    au = arow[u]
                    if au == 0:
                        continue
                    for v in range(n_chunks):
                        bv = brow[v]
                        if bv == 0:
                            continue
                        Acc[i, j, u + v] += au * bv

    # Now reduce the accumulation into n_chunks with carries using base chunk_base
    C = np.zeros((n, n, n_chunks), dtype=np.int64)

    for i in prange(n):
        for j in range(n):
            carry = 0
            # Acc has length 2*n_chunks; we combine and propagate carries into n_chunks buckets.
            for pos in range(0, 2 * n_chunks):
                s = Acc[i, j, pos] + carry
                digit = s % chunk_base
                carry = s // chunk_base
                # only store digits that fit in the target n_chunks window
                if pos < n_chunks:
                    C[i, j, pos] = digit
                else:
                    # when pos >= n_chunks we need to fold digits into higher positions:
                    # add the digit to the correct lower index if within range, else drop (overflow)
                    higher_pos = pos
                    if higher_pos < n_chunks:
                        # (shouldn't normally happen since pos>=n_chunks)
                        C[i, j, higher_pos] = (C[i, j, higher_pos] + digit) % chunk_base
                    else:
                        # overflow beyond available chunks is discarded; choose n_chunks large enough.
                        pass
            # NOTE: any remaining carry after processing positions is discarded (overflow).
    return C


# ---- simple runner / benchmark harness ----
def run_one(n, digits, chunk_digits, tile, trials=3, seed=0):
    """
    n: matrix dimension
    digits: total decimal digits per number (defines magnitude)
    chunk_digits: digits per chunk (chunk base = 10**chunk_digits)
    tile: block tile size (not used in this version of kernel but kept for API compatibility)
    """
    rng = np.random.RandomState(seed)
    # generate random integers with up to `digits` decimal digits (positive)
    max_val = 10 ** digits - 1
    # ensure we don't request a value that exceeds int64 range (guard)
    if max_val > np.iinfo(np.int64).max:
        raise ValueError("Requested 'digits' too large for int64 storage. Reduce digits or chunk size.")

    A = rng.randint(0, max_val + 1, size=(n, n), dtype=np.int64)
    B = rng.randint(0, max_val + 1, size=(n, n), dtype=np.int64)

    n_chunks = (digits + chunk_digits - 1) // chunk_digits
    chunk_base = 10 ** chunk_digits

    # convert to chunks
    A_chunks = int_to_chunks_matrix(A.copy(), chunk_base, n_chunks)
    B_chunks = int_to_chunks_matrix(B.copy(), chunk_base, n_chunks)

    # warm-up JIT with a small block (numba compilation)
    small_n = min(8, n)
    _ = decimal_gemm_numba(A_chunks[:small_n, :small_n, :], B_chunks[:small_n, :small_n, :], chunk_base, tile)

    times = []
    for t in range(trials):
        t0 = time.perf_counter()
        C_chunks = decimal_gemm_numba(A_chunks, B_chunks, chunk_base, tile)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # convert back to integer matrix (may overflow if digits are big)
    C_int = chunks_to_int_matrix(C_chunks, chunk_base)

    return {
        "n": n,
        "digits": digits,
        "chunk_digits": chunk_digits,
        "tile": tile,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "result_sample": int(C_int[0, 0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--digits", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=2,
                        help="decimal digits per chunk (e.g. 2 -> base 100)")
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", type=str, default="numba_decimal_results.csv")
    args = parser.parse_args()

    results = []
    for n in [args.n]:
        print(f"Running decimal numba GEMM n={n} digits={args.digits} chunk={args.chunk} tile={args.tile}")
        r = run_one(n, args.digits, args.chunk, args.tile, trials=args.trials)
        print(" -> mean %.6fs  std %.6fs sample %d" % (r["mean_time"], r["std_time"], r["result_sample"]))
        results.append(r)

    # append results to CSV
    out = Path(args.out)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n", "digits", "chunk_digits", "tile", "mean_time", "std_time", "result_sample"])
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("Saved results to", out)


if __name__ == "__main__":
    main()

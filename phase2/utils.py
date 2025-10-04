"""
phase2.utils
Small utilities used by phase2 (random matrix generator, conversions).
"""

from typing import List
import random

def random_matrix(n: int, digits: int = 8, low: int = 0, high: int | None = None, seed: int | None = None) -> List[List[int]]:
    """
    Return an n x n matrix of random integers (Python int).
    - digits: number of decimal digits (approx range 0 .. 10**digits-1)
    - high: explicit upper bound (overrides digits if given)
    - seed: optional random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    if high is None:
        # avoid huge values that break numpy randint int32 on older code paths
        high = 10 ** digits
    # Make sure high is reasonable (> low)
    if high <= low:
        raise ValueError("high must be > low")
    return [[random.randint(low, high - 1) for _ in range(n)] for _ in range(n)]

def top_left_block(M: List[List[int]], k: int = 2) -> List[List[int]]:
    """Return the k x k top-left block of matrix M (useful for printing)."""
    return [row[:k] for row in M[:k]]

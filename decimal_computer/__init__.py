# decimal_computer/__init__.py
"""
decimal_computer package initializer with defensive imports.

This module imports the submodules and safely pulls a curated public API.
If optional functions are missing (e.g. during incremental edits), imports
won't fail at package import time.
"""
# Core exports
from .context import DecimalContext, get_context
from .fastdecimal import FastDecimal

# Import vector_ops as a module and extract attributes defensively.
from . import vector_ops as _vector_ops

# Helper to safely pull an attribute (returns None if missing)
def _safe(attr_name: str):
    return getattr(_vector_ops, attr_name, None)

# Public API from vector_ops (some may be optional)
mul_vectors = _safe("mul_vectors")
mul_vectors_frozen = _safe("mul_vectors_frozen")
mul_vectors_raw = _safe("mul_vectors_raw")
add_vectors = _safe("add_vectors")
sub_vectors = _safe("sub_vectors")
add_vectors_raw = _safe("add_vectors_raw")
sub_vectors_raw = _safe("sub_vectors_raw")
mul_vectors_view = _safe("mul_vectors_view")
mul_vectors_fast = _safe("mul_vectors_fast")

# GEMM and adaptive components (keep existing exports)
from .decimal_gemm import decimal_gemm_naive
from .decimal_digit_starter import int_to_digits, digits_to_int
from .adaptive import adaptive_gemm

# Build __all__ only with items that are present
__all__ = [
    "DecimalContext",
    "get_context",
    "FastDecimal",
    "decimal_gemm_naive",
    "int_to_digits",
    "digits_to_int",
    "adaptive_gemm",
]

# add vector ops that are present
for name, obj in [
    ("mul_vectors", mul_vectors),
    ("mul_vectors_frozen", mul_vectors_frozen),
    ("mul_vectors_raw", mul_vectors_raw),
    ("add_vectors", add_vectors),
    ("sub_vectors", sub_vectors),
    ("add_vectors_raw", add_vectors_raw),
    ("sub_vectors_raw", sub_vectors_raw),
    ("mul_vectors_view", mul_vectors_view),
    ("mul_vectors_fast", mul_vectors_fast),
]:
    if obj is not None:
        globals()[name] = obj
        __all__.append(name)

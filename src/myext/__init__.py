# Safe package initializer for myext
try:
    from . import gemm as _compiled  # compiled extension (.pyd) if available
except Exception:
    _compiled = None

from . import helpers
gemm = helpers.gemm
__all__ = ["gemm"]

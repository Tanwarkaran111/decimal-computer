# phase4/__init__.py
# Prefer compiled fast FFT implementation if available; register it as submodule alias.
import sys, importlib

_pkg = __name__  # "phase4"
try:
    # try compiled fast module
    _mod = importlib.import_module(_pkg + ".fft_multiply_fast")
    # expose functions at package level
    from .fft_multiply_fast import *
    # register alias so "import phase4.fft_multiply" will load the compiled module
    sys.modules[_pkg + ".fft_multiply"] = _mod
    fft_multiply = _mod
except Exception:
    # fallback to pure-python module
    _mod = importlib.import_module(_pkg + ".fft_multiply")
    from .fft_multiply import *
    sys.modules[_pkg + ".fft_multiply"] = _mod
    fft_multiply = _mod

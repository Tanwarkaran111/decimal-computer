# setup_cython.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

# try to import Cython; if not installed, instruct user
try:
    from Cython.Build import cythonize
except Exception as e:
    raise RuntimeError(
        "Cython is required to build extensions. Install with:\n"
        "    pip install cython\n"
    ) from e

# Optional: include numpy headers if available
include_dirs = []
try:
    import numpy as np
    include_dirs.append(np.get_include())
    has_numpy = True
except Exception:
    has_numpy = False

# Common compiler flags
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    # MSVC-specific flags
    # /Ox = optimize, /EHsc = C++ exceptions model safe for C++
    extra_compile_args = ["/Ox", "/EHsc"]
    # link args left empty by default
else:
    # Unix-ish (gcc/clang)
    extra_compile_args = ["-O3", "-march=native", "-fomit-frame-pointer"]
    extra_link_args = []

# If you'd like to expose macros such as NDEBUG, you can add them here
define_macros = [("PY_SSIZE_T_CLEAN", "1")]

# The extension name should match the python import path you use.
# Here we assume decimal_computer.parallel_cworker (module path).
ext_modules = [
    Extension(
        "decimal_computer.parallel_cworker",
        sources=[
            "decimal_computer/parallel_cworker.pyx",
        ],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )
]

# Compiler directives for cythonize
cython_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
}

# Run cythonize and setup
if __name__ == "__main__":
    # helpful message about numpy inclusion
    if has_numpy:
        print("Building with NumPy include dirs:", include_dirs)
    else:
        print("NumPy not found — building without NumPy headers (still ok).")

    # call setup; this supports commands like:
    #   python setup_cython.py build_ext --inplace
    setup(
        name="decimal_computer",
        ext_modules=cythonize(ext_modules, compiler_directives=cython_directives),
        zip_safe=False,
    )

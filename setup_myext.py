# setup_myext.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os, sys

# ==========================================================
# Source configuration
# ==========================================================

# Explicit C sources used by the gemm extension
c_kernel_sources = [
    "src/kernel_wrapper.c",
    "src/micro_kernel_avx2.c",
    "src/micro_kernel_avx2_opt.c",
    "src/micro_kernel_scalar.c",
]

# The Cython-generated wrapper
pyx_source = "src/myext/gemm.pyx"

# ==========================================================
# Compiler and linker flags
# ==========================================================

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    # Microsoft Visual C++ build flags
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
    print("[setup] Using MSVC flags: /O2 /openmp")
else:
    # Linux / MacOS builds (GCC / Clang)
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
    print("[setup] Using GCC/Clang flags: -O3 -fopenmp")

# ==========================================================
# Extension definition
# ==========================================================

ext = Extension(
    "myext.gemm",
    sources=[pyx_source] + c_kernel_sources,
    include_dirs=[np.get_include(), "include"],
    language="c",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# ==========================================================
# Setup execution
# ==========================================================

setup(
    name="myext_local",
    version="0.1.0",
    description="Local optimized GEMM extension with OpenMP and AVX2 kernels",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize([ext], language_level=3),
    script_args=["build_ext", "--inplace"],
    include_package_data=True,
    package_data={"myext": ["tuned.json"]},
)

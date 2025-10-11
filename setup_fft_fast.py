from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="phase4.fft_multiply_fast",
        sources=["phase4/fft_multiply_fast.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="fft_multiply_fast",
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True},
    ),
)

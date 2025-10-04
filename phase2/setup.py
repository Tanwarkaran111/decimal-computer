from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize(
    "phase2/fast_math.pyx",   # ✅ explicitly inside phase2
    language_level=3
)

setup(
    name="phase2",
    ext_modules=ext_modules,
    packages=["phase2"],
    package_dir={"phase2": "phase2"},
)

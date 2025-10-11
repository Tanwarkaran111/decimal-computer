# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import sys
import os
import tempfile

def env_disable_openmp():
    v = os.environ.get("DECIMAL_COMPUTER_NO_OPENMP", "")
    return v in ("1", "true", "True")

class BuildExtWithOpenMP(build_ext):
    """
    build_ext that checks whether the compiler actually supports OpenMP flags
    before adding them. Respects DECIMAL_COMPUTER_NO_OPENMP env var.
    """
    def has_flag(self, compiler, flagname):
        """
        Try to compile a tiny program with the given flag.
        """
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write('int main() { return 0; }')
            fname = f.name
        try:
            # try compile with flag
            extra_args = [flagname]
            # use the compiler instance to compile the test
            objects = compiler.compile([fname], extra_postargs=extra_args)
        except Exception:
            return False
        finally:
            try:
                os.remove(fname)
            except OSError:
                pass
        return True

    def build_extensions(self):
        if env_disable_openmp():
            # user requested no OpenMP
            build_ext.build_extensions(self)
            return

        compiler = self.compiler

        # Decide flags based on platform & compiler support
        openmp_compile_args = []
        openmp_link_args = []

        if sys.platform.startswith("win"):
            # MSVC: /openmp (supported on modern MSVC)
            if self.has_flag(compiler, '/openmp'):
                openmp_compile_args = ['/openmp']
                # MSVC handles linking for /openmp
        else:
            # try -fopenmp (gcc/clang)
            if self.has_flag(compiler, '-fopenmp'):
                openmp_compile_args = ['-fopenmp']
                openmp_link_args = ['-fopenmp']
            else:
                # try -qopenmp (some AIX/XL compilers) as a fallback
                if self.has_flag(compiler, '-qopenmp'):
                    openmp_compile_args = ['-qopenmp']
                    openmp_link_args = ['-qopenmp']

        # Attach flags to extensions that declared they want OpenMP via marker in extra_compile_args
        for ext in self.extensions:
            # we only add OpenMP to extensions that opt-in (by having 'want_openmp' attr True)
            want = getattr(ext, "want_openmp", False)
            if want and openmp_compile_args:
                ext.extra_compile_args = (ext.extra_compile_args or []) + openmp_compile_args
                ext.extra_link_args = (ext.extra_link_args or []) + openmp_link_args

        build_ext.build_extensions(self)

# Helper: keep backwards-compatible function that was previously used
def _get_openmp_flags_default():
    """
    (Deprecated) kept for compatibility. Prefer compiler detection via BuildExtWithOpenMP.
    """
    if env_disable_openmp():
        return [], []
    if sys.platform.startswith("win"):
        return ["/openmp"], []
    else:
        return ["-fopenmp"], ["-fopenmp"]

# === your extension definitions (preserve names & sources) ===
ext_modules = [
    Extension(
        "decimal_computer._c_mul",
        ["decimal_computer/_c_mul.pyx"],
        language="c++",
        # we do NOT want to forcibly add OpenMP to _c_mul (keeps it plain C++)
        # if you later need OpenMP for this module set want_openmp=True
        # e.g. Extension(..., want_openmp=True)
    ),
    Extension(
        "decimal_computer._c_quantize",
        ["decimal_computer/_c_quantize.pyx"],
        language="c++",
        # mark that this extension *wants* OpenMP; BuildExtWithOpenMP will only add flags
        # if the compiler supports them and the env var doesn't disable OpenMP.
        extra_compile_args=[],
        extra_link_args=[],
    ),
]

# Mark the second ext to opt-in to OpenMP
# (we can't pass want_openmp as kwarg to Extension on some setuptools versions,
#  so attach attribute after creation)
for ext in ext_modules:
    if ext.name.endswith("_c_quantize"):
        setattr(ext, "want_openmp", True)

# Add our AVX2 GEMM micro-kernel extension
gemm_ext = Extension(
    "fast_gemm._native",
    sources=[
        "src/micro_kernel_scalar.c",
        "src/micro_kernel_avx2.c",
        "src/kernel_wrapper.c",
    ],
    include_dirs=["include"],
    extra_compile_args=["-O3", "-mavx2", "-mfma", "-std=c11"],
    extra_link_args=[],
)

# Append to existing list
ext_modules.append(gemm_ext)

# Cythonize (keep language_level=3)
ext_modules = cythonize(ext_modules, language_level=3)

setup(
    name="decimal_computer_c_ext",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtWithOpenMP},
)

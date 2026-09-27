"""Portable C build by default; explicitly opt into machine-specific instructions."""
import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

openmp = os.environ.get("MYMF_OPENMP", "auto")
native = os.environ.get("MYMF_NATIVE", "0") == "1"
if sys.platform == "win32":
    compile_args = ["/O2"]
    link_args = []
    if openmp != "0":
        compile_args += ["/openmp"]
elif sys.platform == "darwin":
    compile_args = ["-O3", "-std=c11"]
    link_args = []
    if openmp == "1":
        compile_args += ["-Xpreprocessor", "-fopenmp"]
        link_args += ["-lomp"]
    if native:
        compile_args += ["-march=native"]
else:
    compile_args = ["-O3", "-std=c11"]
    link_args = []
    if openmp != "0":
        compile_args += ["-fopenmp"]
        link_args += ["-fopenmp"]
    if native:
        compile_args += ["-march=native"]

extension = Extension(
    "mymf._kernels",
    ["src/mymf/_kernels.pyx"],
    include_dirs=[str(Path("src/mymf"))],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=[("MYMF_NATIVE_BUILD", "1" if native else "0")],
    language="c",
)
setup(ext_modules=cythonize(
    [extension], compiler_directives={"language_level": 3},
))

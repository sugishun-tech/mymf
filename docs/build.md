# Building and platform notes

## Normal install

```bash
python -m pip install .
python -c 'from mymf import MyMF, build_info; print(build_info())'
```

Python 3.10+ is required. Build dependencies are setuptools >=77, wheel and
Cython >=3.0. Runtime dependencies are in pyproject.toml. The extension uses
memoryviews, not the NumPy C API, so NumPy headers are not a build dependency.
A C compiler and Python development headers are needed. Linux defaults to GCC
with `-O3 -std=c11 -fopenmp`; only C, not C++, is compiled.

## Resource-constrained or offline builds

With compatible build/runtime dependencies already installed:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" python -m pip install --no-cache-dir \
  --no-build-isolation --no-deps .
```

`--no-deps` does not install missing dependencies; ensure they are present first.
The local release validation used `--no-build-isolation` because the execution
environment could not resolve external package/dataset hosts. Default isolated
builds are defined by pyproject.toml, but downloading dependencies was not tested
in this environment.

## Controls

| Environment variable | Build/run time | Effect |
|---|---|---|
| `MYMF_OPENMP=0` | Build | Compile serial kernels. `n_jobs` cannot create parallelism in this build. |
| `MYMF_OPENMP=1` | Build | Explicitly enable OpenMP, including macOS. |
| `MYMF_NATIVE=1` | Build | Add `-march=native` on supported GCC/Clang platforms. Do not distribute that binary to unlike CPUs. |
| `MYMF_SIMD=baseline` | Before Python import | Disable runtime x86 SIMD dispatch. |
| `MYMF_SIMD=avx2` | Before Python import | Prefer AVX2/FMA, with safe baseline fallback if unsupported. |
| `MYMF_SIMD=auto` or unset | Before Python import | Prefer supported AVX-512/FMA, then AVX2/FMA, then baseline. |

`MYMF_NATIVE` is not a supported MSVC tuning switch. The default portable build
uses runtime function dispatch on supported Linux x86-64 compilers; it does not
require all target machines to support AVX-512. A Linux wheel still has Python,
platform, libc and libgomp dependencies: runtime SIMD dispatch does **not** make it
a universally portable wheel. Use standard wheel auditing/repair before public
binary distribution.

When changing build flags, remove stale generated/build output first:

```bash
rm -rf build src/mymf/_kernels.c src/mymf/_kernels*.so src/mymf/_kernels*.pyd
MYMF_OPENMP=0 python -m pip install --no-build-isolation --no-deps --force-reinstall .
```

## Platforms

Linux x86-64/GCC/OpenMP and a separate serial build were locally compiled and tested
on CPython 3.13. macOS defaults to serial because Apple Clang does not bundle
OpenMP. Explicit OpenMP needs a working libomp installation and compiler include/
library search flags. Windows uses MSVC `/O2` and `/openmp` by default; use
`MYMF_OPENMP=0` for serial. Additional-platform CI configurations are supplied,
but macOS, Windows, ARM and other Python versions were not locally executed.

## Build distributions

```bash
python -m pip install build
python -m build
```

A Cython-generated C file is intentionally not versioned or included in the source
ZIP. The build generates it from `_kernels.pyx`; `_simd.h` is part of the sdist.
Locally, wheel and sdist production were also checked using the installed build
backend without downloading the optional `build` frontend.

## BLAS and concurrent work

Use `n_jobs` for sparse kernels and `blas_threads` for BLAS. The default BLAS limit
is one; it is often reasonable for skinny sparse-workload matrices but not a
universal optimum. `n_jobs=None` respects joblib's CPU-quota-aware count and caps it
at eight; it does not blindly launch all host hardware threads.

For outer joblib multiprocessing, use `n_jobs=1` per model unless intentionally
partitioning the core budget. For concurrent Python threads, set external BLAS
limits and use `blas_threads=None` to avoid process-wide threadpoolctl context
interference. Benchmark on the deployed CPU, compiler and sparsity pattern.

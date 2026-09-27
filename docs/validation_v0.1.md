# Historical validation record: v0.1.0

Release 0.1.0, validated 2026-09-27 on Linux x86-64, CPython 3.13.5.
NumPy 2.3.5, SciPy 1.17.0, sklearn 1.8.0, Cython 3.2.4, GCC 14.2.0.

| Configuration | Result |
|---|---|
| Default OpenMP build, automatic AVX-512/FMA selection | 141 passed |
| Same OpenMP build, MYMF_SIMD=avx2 | 141 passed |
| Same OpenMP build, MYMF_SIMD=baseline | 141 passed |
| Independently compiled MYMF_OPENMP=0 serial build | 141 passed |
| Wheel built from an extracted source distribution and installed separately | 141 passed |
| Installed wheel import and fit/transform smoke test | Passed; extension came from the wheel's installation path |
| Runnable basic, recommendation and Pipeline examples | Passed |

The count is parameterized pytest cases, not independent algorithms or datasets.
It includes sklearn's check_estimator with MyMF(n_jobs=1), pandas feature validation,
set_output, native index/dtype variants, controlled independent SVD-reference
comparisons, malformed CSR rejection and degenerate matrices. Pandas is optional;
that case skips if pandas is absent. The supplied CI explicitly installs it.

## Packaging check

A clean source copy without generated C or shared libraries produced an sdist.
The sdist was extracted, its Cython extension compiled, and a wheel built using
pip's PEP 517 path with --no-build-isolation --no-deps. That wheel was installed
in a separate virtual environment. Both mymf.__file__ and _kernels.__file__ were
asserted to be under the wheel installation, not the development source tree.
The tests were run from outside the source repository.

For this offline check, already installed runtime dependencies were exposed to the
separate environment via an explicit .pth file. This validates the wheel and native
module installation, but is **not** a fresh, networked dependency-installation test.
No claim is made that isolated dependency downloads succeeded.

The distributed ZIP intentionally contains source, not the local CPython 3.13
Linux wheel, build products or downloaded datasets. Users compile for their own
Python and platform. Remote CI jobs, macOS, Windows, ARM and other Python versions
have not been executed locally; their configuration is not evidence of success.

Raw short test logs and source-file SHA-256 values are in
`benchmarks/results/validation*.txt` and `benchmarks/results/validation.json`.
The benchmark protocol and limitations are separately documented in benchmarks.md.

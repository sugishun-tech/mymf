# Local validation record: v0.2.0

Linux x86-64, AMD EPYC 9V74, CPython 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
sklearn 1.8.0, Cython 3.2.4, GCC 14.2.0. Portable compilation; no `-march=native`.

| Configuration | Result |
|---|---|
| Default OpenMP build, automatic AVX-512/FMA selection | 291 passed |
| Same OpenMP build, `MYMF_SIMD=avx2` | 291 passed |
| Same OpenMP build, `MYMF_SIMD=baseline` | 291 passed |
| Independently compiled `MYMF_OPENMP=0` serial build | 291 passed |
| Wheel built from an extracted source distribution and installed separately | 291 passed |
| Installed wheel import and fit/transform smoke test | Passed; Python module and native extension both resolved inside the separate installation |
| Runnable basic, recommendation and Pipeline examples against that wheel | Passed |

The count is parameterized pytest cases, not 291 independent algorithms or datasets.
The tests include sklearn's `check_estimator`, pandas feature validation, `set_output`,
serialization, independent SVD-reference comparisons, and malformed/degenerate input.
Added cases cover SIMD vector boundaries and masked tails, all sparse index/real dtype
combinations, read-only input, serial/parallel output agreement, Welford variance against
a dense reference, random-stream behavior, thread-limit restoration, and changed inputs
being recomputed rather than served from cached factors. Pandas is optional and its
case skips without pandas; it was present here and is explicitly installed in CI.

## Packaging check

A clean source copy without generated C or shared libraries produced an sdist using
`python setup.py sdist`. The sdist was extracted, its Cython extension compiled, and a
wheel built using pip's PEP 517 path with `--no-build-isolation --no-deps`. That wheel
was installed in a separate virtual environment. Both `mymf.__file__` and
`_kernels.__file__` were asserted to be under that installation, not the development
tree. Tests were run from outside the source repository.

The offline virtual environment exposes the already installed runtime dependencies
through an explicit `.pth` path. An initial attempt using only `--system-site-packages`
did not expose the parent virtual environment's dependencies; no tests passed in that
initial setup. The successful installation and 291-case run used the explicit path.
This checks source packaging, native compilation and wheel installation, but is **not**
a fresh networked dependency-download test.

The distributed ZIP contains source, not local CPython-specific binaries or datasets.
Remote CI jobs, macOS, Windows, ARM and other Python versions were not executed locally.
AVX2 and baseline overrides used the same physical AVX-512-capable CPU; these do not
constitute tests on different processors. No sanitizer run is claimed.

Raw logs are in `benchmarks/results/v0.2/validation_*.txt`. The accompanying
`validation.json` records source-file hashes and the local wheel hash. That local
wheel is not redistributed. The Alice sample was preserved unchanged from the supplied
article; no result on the actual Alice text is claimed without that input file.

For measured runtime and recommendation quality, see [benchmarks.md](benchmarks.md).
The historical v0.1 report is [validation_v0.1.md](validation_v0.1.md).

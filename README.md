# mymf

**CPU-first approximate matrix factorization with a scikit-learn-style API.**

`MyMF` implements dynamically shifted randomized subspace iteration, based on
[dashSVD, Algorithm 1043 (2024)](https://doi.org/10.1145/3660629), with Cython/C
sparse kernels, OpenMP and runtime SIMD dispatch. It prioritizes the speed/accuracy
trade-off of low-rank sparse problems. Dense products use NumPy/SciPy BLAS.

It is an independently written implementation, not a wrapper around the authors'
C/MKL library. Neither the algorithm itself nor a universal speed record is claimed
as new. See [research and scope](docs/research.md).

## Install from this repository

Python 3.10+ and a C compiler are required. The compiled extension is mandatory;
there is no silent Python fallback. No C++, CUDA, GPU or MKL dependency is required.

```bash
# Debian: install compiler and Python development headers when needed.
sudo apt-get install build-essential python3-dev python3-venv

cd mymf
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

This source release has **not** been published to PyPI; use `pip install .`, not
`pip install mymf`. The default Linux build enables OpenMP. macOS defaults to serial;
`MYMF_OPENMP=0 python -m pip install .` explicitly requests a serial build. See
[build notes](docs/build.md) for offline environments, platform support and tuning.

## Quick start

```python
import numpy as np
from scipy import sparse
from mymf import MyMF, build_info

X = sparse.random(10_000, 5_000, density=0.002, format="csr",
                  dtype=np.float32, random_state=42)
mf = MyMF(n_components=64, n_iter=2, random_state=42, n_jobs=-1)
Z = mf.fit_transform(X)           # (10000, 64), dense
Z_new = mf.transform(X[:10])      # (10, 64), same feature space
print(mf.components_.shape)       # (64, 5000)
print(build_info())              # Actual OpenMP/SIMD configuration

# Only reconstruct a small batch: inverse_transform always returns dense output.
X_approx = mf.inverse_transform(Z[:10])
item_ids, scores = mf.recommend(X[:10], n_recommendations=20)
```

`fit_transform(X)` returns the projection `X @ components_.T` up to floating-point
roundoff, not an incompatible training-only approximation. `fit`, `transform`,
`inverse_transform`, cloning, parameter search, Pipeline, feature names and
`set_output` are supported. This is **API similarity**, not a byte-for-byte
replacement for every `TruncatedSVD` option. See [API reference](docs/api.md).

## What problem does it solve?

For a non-centred matrix X, it finds approximately orthonormal rows C such that
`X ≈ (X @ C.T) @ C`, with rank at most `n_components`.

In a user–item matrix, implicit zeros participate in the reconstruction objective.
This is **not** observed-only rating regression, confidence-weighted implicit ALS,
BPR, nonnegative factorization or matrix completion with a missing-value mask.
Reconstruction quality and ranking quality are different objectives. Evaluate
recommendation quality on held-out interactions before deployment.

## Measured example, not a universal guarantee

Synthetic 50,000 × 20,000 float32 CSR matrix, 2,444,696 nonzeros, rank 64,
oversampling 10, both solvers single-threaded including BLAS, three seeds.
Time is the median complete `fit_transform`; error is the mean relative Frobenius
reconstruction error. Smaller is better in both columns.

| Model | Iterations | Seconds | Relative error |
|---|---:|---:|---:|
| sklearn TruncatedSVD | 2 | 0.757676 | 0.975356 |
| MyMF | 2 | 0.491231 | 0.975343 |
| sklearn TruncatedSVD | 5 | 1.289154 | 0.975190 |
| MyMF | 4 | 0.675301 | 0.975171 |

In this measurement, MyMF q=2 is **1.54× faster at the same iteration count**;
MyMF q=4 is **1.91× faster than sklearn q=5** with marginally lower measured
reconstruction error. The default-vs-default q=2 vs q=5 speed ratio is 2.62×,
but q=2 is less accurate than sklearn q=5. Those are different comparisons.

Dense input showed much smaller gains; q=2 had slightly worse reconstruction
error than sklearn q=2. Fresh-process peak RSS on the sparse case was **265.56 MiB
for MyMF vs 250.86 MiB for sklearn**, so lower memory than sklearn is not claimed.
All recommendation results supplied here are **synthetic**, not MovieLens results.
The original dashSVD/MKL implementation and other SOTA libraries were not timed.

CPU: reported Intel Xeon Platinum 8573C, Linux, four-CPU cgroup quota; Python
3.13.5, NumPy 2.3.5, SciPy 1.17.0, sklearn 1.8.0, Cython 3.2.4, GCC 14.2.0,
OpenBLAS 0.3.30. Portable compilation with runtime AVX-512/FMA was selected.
[Methodology, all measurements and limitations](docs/benchmarks.md) ·
[Raw JSON](benchmarks/results/)

## Tune for your workload

Start with float32 CSR, `n_iter=2`, `n_oversamples=10`, and an explicit `random_state`.
Compare q=0, 2 and 4 on held-out data; **q=0 can lose substantial accuracy**. Raise
`n_iter` or `n_oversamples` if needed. Use `dtype="float64", normalizer="qr"` for
more difficult numerical problems; it costs more. `tol` is an optional heuristic,
not a certified residual tolerance, and is disabled by default.

`n_jobs` controls sparse OpenMP threads, while `blas_threads` separately controls
BLAS. Start with BLAS=1 for sparse workloads and benchmark 1/2/4 OpenMP threads.
Dense workloads may benefit from higher `blas_threads`. AVX-512 is not always
faster: compare `MYMF_SIMD=avx2` and `MYMF_SIMD=baseline` in fresh Python processes.

## Test and reproduce

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/basic.py
python examples/recommendation.py
python benchmarks/run.py --case sparse_large --threads 1 --repeat 3 \
  --output benchmarks/results/local_sparse.json
python benchmarks/recommendation.py --output benchmarks/results/local_recommendation.json
```

The release was tested on Linux CPython 3.13 with 141 tests, including sklearn's
`check_estimator`, actual compiled-extension checks and independent mathematical
reference comparisons. CI definitions are supplied for additional platforms;
these do not imply those remote jobs have already run. See `docs/validation.md`
for the exact local validation record.

## Repository

```text
src/mymf/          Estimator, Cython kernels, C SIMD dispatch and small-matrix algebra
examples/          Runnable embedding, recommendation and Pipeline examples
benchmarks/        Offline timing, quality, recommendation and peak-RSS measurements
tests/             Numerical, API, native-kernel and edge-case tests
docs/              API, algorithm, build, research and reproducibility notes
.github/workflows/ Linux, macOS-serial and Windows build/test CI definitions
```

## License and attribution

MIT. Algorithm attribution: Xu Feng, Wenjian Yu, Yuyang Xie and Jie Tang,
*Algorithm 1043: Faster Randomized SVD with Dynamic Shifts*, ACM TOMS (2024),
DOI 10.1145/3660629. Bibliographic comparisons are in `docs/research.md`.

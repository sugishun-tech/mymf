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

## MovieLens 100K performance: v0.2.0

Measured on the supplied real `u.data`: float32, rank 32, oversampling 10,
one sparse thread and one BLAS thread. 10 seeds x 30 fresh fits per method.
Complete `fit_transform` is timed, including validation, random sketches, transpose
construction, projection and variance. See [the protocol and all results](docs/benchmarks.md).

| Method | Median ms | Recall@20 | NDCG@20 | Relative Frobenius error |
|---|---:|---:|---:|---:|
| TruncatedSVD_q2 | 5.647 | 0.195720 | 0.158065 | 0.734004 |
| MyMF_q2 | 3.451 | 0.195770 | 0.158434 | 0.731105 |
| TruncatedSVD_q5 | 9.448 | 0.197715 | 0.160791 | 0.728483 |
| MyMF_q5 | 5.018 | 0.198344 | 0.161064 | 0.727808 |
| MyMF_q4 | 4.512 | 0.197540 | 0.160583 | 0.728152 |
| MyMF_q0 | 2.375 | 0.177086 | 0.143387 | 0.759070 |

At the **same iteration count**, MyMF is **1.64x** faster at q=2 and **1.88x**
faster at q=5 in this run. The small metric differences are not a statistical
superiority claim. Fresh-process first fits and AVX2 dispatch were also measured.
MyMF does not call sklearn's SVD, recognize MovieLens, cache fitted factors,
reduce the requested rank, or silently stop iterations early.

v0.2 reduces repeated library-discovery overhead, uses register-blocked masked-tail
SIMD kernels, bypasses serial OpenMP regions, avoids duplicate CSR validation,
generates float32 Gaussian sketches directly, and computes stable sparse variance
without copying all values to float64. The shifted SVD algorithm is unchanged.

**Seed compatibility:** integer seeds use PCG64 in v0.2 and produce different
factors from v0.1. Pass `np.random.RandomState(seed)` for the legacy Gaussian stream.
`None` still respects `np.random.seed`; `np.random.Generator` is now accepted.
A cached native-library controller does not automatically see libraries loaded later;
call `mymf.refresh_threadpools()` outside concurrent calls after loading new BLAS libraries.

No benchmark guarantees speed on every machine. Historical v0.1 measurements are
kept separately rather than silently relabelled as current results.

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

The release was tested on Linux CPython 3.13 with 291 tests, including sklearn's
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

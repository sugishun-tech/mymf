# Benchmark report: v0.2.0

## MovieLens 100K: actual uploaded data, not synthetic

Input SHA-256: `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`. The file is the supplied official
`ml-100k/u.data`. Dataset bytes are deliberately not redistributed in this project.
The split code is unchanged from v0.1: ratings >= 4 are positives; the latest 20%
of each user's positives are held out if the user has at least five positives.
943 users x 1,682 items; 43,933 training positives; 11,442 test positives; 938 evaluated users.
Candidates include every item except the user's training positives. No negative sampling.
The per-user timestamp split is NOT a globally time-consistent split across all users.
Scores and stable item-ID tie-breaking are evaluated by the same function for both models.

Linux x86-64, AMD EPYC 9V74, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
sklearn **1.8.0** (not a claim about the latest release), Cython 3.2.4, OpenBLAS 0.3.30.
Portable OpenMP build without `-march=native`; runtime AVX-512/FMA selected.
Both BLAS and MyMF sparse kernels use **one thread**. Float32, rank 32, oversampling 10,
`tol=0`. No reduced rank, omitted statistics, or early stopping is used for the speed comparison.

## Fresh-fit timing protocol

10 sketch seeds (0..9), 30 fresh models per seed and method = **300 fits per method**.
All exact methods are warmed once; method order is randomized in each timing round.
The complete `fit_transform` includes input checks, CSR preparation, random numbers,
transpose construction, factorization, public projection, variance, and internal BLAS limits.
Only imports, dataset loading, model construction, warmup, and metric evaluation are excluded.
Every timing, including slow samples, is saved. The reported median pools all 300 fits;
quality metrics average the 10 seeds, not 300 repetitions of the same random draws.
No input, factor, decomposition, or transpose is cached across fits.
Only native-library discovery is reused by the thread-limit controller.

| Method | Median ms | Recall@20 | NDCG@20 | Relative Frobenius error |
|---|---:|---:|---:|---:|
| TruncatedSVD_q2 | 5.647 | 0.195720 | 0.158065 | 0.734004 |
| MyMF_q2 | 3.451 | 0.195770 | 0.158434 | 0.731105 |
| TruncatedSVD_q5 | 9.448 | 0.197715 | 0.160791 | 0.728483 |
| MyMF_q5 | 5.018 | 0.198344 | 0.161064 | 0.727808 |
| MyMF_q4 | 4.512 | 0.197540 | 0.160583 | 0.728152 |
| MyMF_q0 | 2.375 | 0.177086 | 0.143387 | 0.759070 |

Same-q speedups are **1.636x** at q=2 and
**1.883x** at q=5. Mean ranking metrics do not
fall below sklearn in this run, but tiny differences are not evidence of statistically
superior recommendation quality. q=0 is deliberately shown as a poor quality/speed tradeoff.
The reconstruction objective is still zero-filled unweighted SVD, not implicit ALS or BPR.

## Random stream compatibility and ablation

v0.2 integer seeds use NumPy PCG64. Float32 sketches are generated directly in float32.
The sketch is still Gaussian and its dimensions have not changed. Identical seed numbers
are not identical sketches between MyMF versions or between MyMF and sklearn.
Explicit `np.random.RandomState(seed)` retains the old Gaussian stream; `None` still
respects `np.random.seed`. A `Generator` instance is also accepted and advanced in place.

The optional legacy-stream measurements use the same v0.2 implementation with an
explicit RandomState object. They give 3.955 ms
at q=2 and 5.535 ms at q=5, also faster than
the paired sklearn timings. The RandomState object's construction is outside the fit timer,
just as estimator construction is. This is not a bitwise or timing reproduction of v0.1.

## CPU dispatch and first-fit checks

An AVX2-only run uses the same CPU with `MYMF_SIMD=avx2`, five seeds and 15 fits per seed.
It is a check of the alternate kernel, not a benchmark of an older physical CPU.
The raw file is `benchmarks/results/v0.2/movielens_avx2_t1.json`.

A separate experiment launches a new interpreter for every sample, with **no MyMF warmup**,
five processes per method. It includes initial controller discovery and LAPACK first use,
but excludes Python imports and dataset loading for both models:

| Method | First-fit median ms |
|---|---:|
| sklearn_q2 | 6.673 |
| sklearn_q5 | 10.903 |
| mymf_q2 | 4.922 |
| mymf_q5 | 6.370 |

## Large sparse regression check

Same synthetic generator as v0.1; 50,000 x 20,000; 2,444,696 stored values;
float32; rank 64; oversampling 10; 3 seeds; one BLAS and one sparse thread.
These are new v0.2 measurements, not relabelled v0.1 numbers.

| Method | Median seconds | Relative Frobenius error |
|---|---:|---:|
| sklearn_q2 | 0.447861 | 0.97535604 |
| mymf_q2 | 0.219736 | 0.97534519 |
| sklearn_q5 | 0.767076 | 0.97519049 |
| mymf_q4 | 0.319305 | 0.97517225 |
| mymf_q5 | 0.349339 | 0.97513339 |

No new dense or peak-memory advantage is claimed. The v0.1 dense, memory, and
synthetic-recommendation results are retained explicitly as history in
[benchmarks_v0.1.md](benchmarks_v0.1.md), with original JSON files at `benchmarks/results/`.
They are not measurements of v0.2. The user's v0.1 MovieLens result is retained under
`benchmarks/results/v0.1/user_local_movielens.json`; its machine is different/unspecified,
so it must not be used to calculate an end-to-end before/after speedup on this machine.

## Reproduce

```bash
python benchmarks/recommendation.py \
  --movielens /path/to/ml-100k/u.data --threads 1 \
  --rank 32 --oversamples 10 --repeat 10 --timing-repeats 30 \
  --include-legacy --assert-faster \
  --output benchmarks/results/local_movielens.json

MYMF_SIMD=avx2 python benchmarks/recommendation.py \
  --movielens /path/to/ml-100k/u.data --threads 1 \
  --repeat 5 --timing-repeats 15 --assert-faster \
  --output benchmarks/results/local_movielens_avx2.json

python benchmarks/cold_start.py --movielens /path/to/ml-100k/u.data \
  --repeat 5 --output benchmarks/results/local_cold.json

python benchmarks/run.py --case sparse_large --threads 1 --repeat 3 \
  --output benchmarks/results/local_large.json
```

`--assert-faster` saves all results before raising if q=2 or q=5 fails to beat sklearn
on the local machine. It does not adjust parameters, rerun until successful, or discard
slow fits. The benchmark is an offline command, not a noisy CI unit test.
CPU, compiler, BLAS, thread quotas and data can change the result; no universal claim
is made for every machine, rank, matrix or competing SVD implementation.

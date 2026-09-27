# Historical benchmark report: v0.1.0

## Scope and environment

All bundled numeric results were actually run locally for this release. All data
in those results are **synthetic**. There is no MovieLens or production evaluation.
The baseline is the **installed sklearn 1.8.0**, not a claim to have timed the newest
online release. The original dashSVD/MKL implementation, PRIMME_SVDS, block Krylov,
PCAone and Row-aware Randomized SVD were not timed.

Reported CPU: Intel Xeon Platinum 8573C. The Linux cgroup quota is four CPU cores
(`cpu.max: 400000 100000`); host/affinity counts are not assumed to be available
compute budget. Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, sklearn 1.8.0,
Cython 3.2.4, GCC 14.2.0 and OpenBLAS 0.3.30. Default portable compilation was
used, with OpenMP enabled and runtime AVX-512/FMA selected. No GPU, -march=native,
-ffast-math or external dataset cache was used.

## Timing protocol

`benchmarks/run.py` warms up each method once, changes method order across repetitions,
uses fresh estimators with the same seed numbers, and measures complete
`fit_transform` using `perf_counter`. Input validation, CSR canonicalization,
transpose construction, sketch generation, iteration, factor extraction, embedding
and explained-variance statistics are inside the timer. Compilation, data generation,
explicit garbage collection and quality evaluation are outside it. Reported times
are medians; the JSON also contains each observation, minima, maxima and seeds.
There are three repetitions for the large case, five for medium and dense.

These are warm-run in-memory measurements, not cold filesystem loads or total
pipeline latency. Identical random seeds do not imply identical randomized sketches
between different algorithms/orientations. Rank is 64 and oversampling is 10 for
all methods. All MyMF runs use `normalizer="auto"`, `tol=0`, default orientation,
and input-preserving dtype. A shift-disabled MyMF is included as a controlled
ablation; it is not a second independent implementation of sklearn.

## Quality protocol

Let `C = components_`, `Z = X @ C.T`, all evaluated in float64. We measure
`||X - Z C||_F / ||X||_F` without constructing the dense reconstruction, using

```text
residual² = ||X||_F² - 2 trace(Z.T Z) + sum((Z.T Z) * (C C.T))
```

The last term retains the actual loss of component orthogonality. Projection
energy `||Z||_F² / ||X||_F²` is also reported, but it is not identical to sklearn's
centred explained-variance ratio. The raw records include orthogonality error.
Sparse cases have no computed exact best rank-64 reference. Dense data have a
known spectrum, so the optimum error and leading singular-value error are available.
Three/five seeds are descriptive measurements, not a significance study.

## Large sparse, one thread

50,000 × 20,000 float32 CSR, 2,444,696 nonzeros, 19,757,572 input-storage bytes.
Each row is generated from 50 events: a random one of 32 taste groups, with 75%
within-group draws and 25% uniform-noise draws; duplicates become event counts.
Both methods and BLAS are single-threaded. This is the primary apples-to-apples
single-core timing comparison.

| Model | Median seconds | Projection energy | Relative Frobenius error |
|---|---:|---:|---:|
| sklearn_q2 | 0.757676 | 0.04868061 | 0.97535604 |
| sklearn_q5 | 1.289154 | 0.04900352 | 0.97519049 |
| mymf_q0 | 0.320158 | 0.00840239 | 0.99578994 |
| mymf_q2 | 0.491231 | 0.04870590 | 0.97534307 |
| mymf_q4 | 0.675301 | 0.04904119 | 0.97517117 |
| mymf_q5 | 0.772221 | 0.04911607 | 0.97513278 |
| mymf_unshifted_q2 | 0.497135 | 0.04856575 | 0.97541491 |

Same q=2: sklearn/MyMF time ratio is
**1.54×**.
Default q=5 vs MyMF q=2 is
**2.62×**, but with a
slightly worse reconstruction error for MyMF q=2. MyMF q=4 vs sklearn q=5 is
**1.91×** and has marginally
lower measured error on this particular dataset.

The relative error around 0.975 is large: rank 64 captures only about 4.9% of
uncentred squared energy on this noisy matrix. This is not "97.5% accuracy".
The purpose is to compare like-for-like low-rank approximation, not to present
this data model as nearly lossless. Error differences in the last digits are
small in practical terms and do not imply better ranking performance.

## Large sparse, four MyMF workers

| Model | Median seconds | Projection energy | Relative Frobenius error |
|---|---:|---:|---:|
| sklearn_q2 | 0.762277 | 0.04868061 | 0.97535604 |
| sklearn_q5 | 1.349100 | 0.04900352 | 0.97519049 |
| mymf_q0 | 0.253986 | 0.00840239 | 0.99578994 |
| mymf_q2 | 0.343100 | 0.04870590 | 0.97534307 |
| mymf_q4 | 0.423346 | 0.04904119 | 0.97517117 |
| mymf_q5 | 0.456147 | 0.04911607 | 0.97513278 |
| mymf_unshifted_q2 | 0.336815 | 0.04856575 | 0.97541491 |

MyMF's sparse products use four OpenMP workers; BLAS is limited to one for **both**
methods. sklearn's sparse product path in this benchmark does not use that OpenMP
parallelism. Therefore this is **not a same-utilized-thread comparison**. Default
sklearn q=5 / MyMF q=2 is 3.93×.
MyMF q=2 itself improves by only
1.43× from its one-worker run.
Do not describe this as a fourfold parallel speedup.

## Medium sparse, one thread

20,000 × 8,000 float32 CSR, 765,127 nonzeros, 6,201,020 input-storage bytes.
Same generation rule with 40 events per row, five measured seeds.

| Model | Median seconds | Projection energy | Relative Frobenius error |
|---|---:|---:|---:|
| sklearn_q2 | 0.211701 | 0.09579179 | 0.95089864 |
| sklearn_q5 | 0.331251 | 0.09656657 | 0.95049116 |
| mymf_q0 | 0.096108 | 0.02452992 | 0.98765888 |
| mymf_q2 | 0.150041 | 0.09584488 | 0.95087072 |
| mymf_q4 | 0.200782 | 0.09664834 | 0.95044813 |
| mymf_q5 | 0.232561 | 0.09682848 | 0.95035336 |
| mymf_unshifted_q2 | 0.146554 | 0.09552351 | 0.95103969 |

The corresponding four-worker run is in `sparse_medium_t4.json`, with the same
thread-allocation caveat as the large case.

## Dense, one thread

1,200 × 600 float64; orthogonal factors with known singular values geometrically
spaced from 50 down to 0.05. Five measured seeds. Dense products use BLAS.

| Model | Median seconds | Projection energy | Relative Frobenius error |
|---|---:|---:|---:|
| sklearn_q2 | 0.033950 | 0.76787383 | 0.48179463 |
| sklearn_q5 | 0.054655 | 0.77119651 | 0.47833407 |
| mymf_q0 | 0.015136 | 0.59838865 | 0.63371894 |
| mymf_q2 | 0.031549 | 0.76656809 | 0.48314787 |
| mymf_q4 | 0.043417 | 0.77103980 | 0.47849785 |
| mymf_q5 | 0.049244 | 0.77132240 | 0.47820246 |
| mymf_unshifted_q2 | 0.029028 | 0.76502868 | 0.48473833 |

MyMF q=2 is only about 1.08× faster than sklearn q=2 and has slightly higher
reconstruction error. MyMF q=4 also has slightly higher error than sklearn q=5.
MyMF q=5 has slightly lower error than sklearn q=5 in this example. There is no
basis for extrapolating the sparse speed ratios to dense matrices. JSON records
also report error divided by the known optimal rank-64 error.

## Synthetic recommendation

2,400 users × 1,600 items, 20 groups. Each user has 40 distinct preferred-group
positives and 10 distinct positives from outside the group, randomly ordered.
A per-user 20% holdout gives 96,000 training positives and 24,000 held-out positives.
Synthetic timestamps only randomize the split; they do not model real time.

Rank 32, oversampling 10, seeds 0/1/2, one OpenMP thread and one BLAS thread.
All items except training positives are candidates, with **no sampled negatives**.
An independent common scorer computes `transform(query) @ components_` for both
solvers, with stable item-ID tie-breaking. Its ranking metrics are not computed
through MyMF's recommendation helper, preventing helper-specific advantages.
Timing covers fitting only, not recommendation/evaluation.

| Model | Median fit seconds | Recall@20 (mean ± std) | NDCG@20 (mean ± std) |
|---|---:|---:|---:|
| sklearn_q2 | 0.017261 | 0.329028 ± 0.000761 | 0.252977 ± 0.000596 |
| sklearn_q5 | 0.025017 | 0.328250 ± 0.001604 | 0.251680 ± 0.000658 |
| mymf_q0 | 0.009081 | 0.318333 ± 0.001715 | 0.248112 ± 0.001247 |
| mymf_q2 | 0.015026 | 0.329472 ± 0.000255 | 0.253213 ± 0.000406 |
| mymf_q4 | 0.018262 | 0.330542 ± 0.000932 | 0.253755 ± 0.001250 |
| mymf_q5 | 0.022108 | 0.329611 ± 0.000554 | 0.252860 ± 0.001136 |

These differences are small relative to the tiny number of tested seeds. They do
not establish better real-world recommendation quality. Reconstruction error and
Recall/NDCG need not move together. See `recommendation_synthetic.json` for the
complete observations and standard deviations.

### Optional real-data run

Obtain the official [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)
and respect its usage terms. Supply the local `u.data` file:

```bash
python benchmarks/recommendation.py --movielens /path/to/ml-100k/u.data \
  --repeat 3 --threads 1 --output benchmarks/results/local_movielens.json
```

This uses ratings >=4 as positives, holds out each eligible user's latest 20%,
evaluates users with at least five positives and ranks the full catalogue after
excluding training positives. A source-file SHA-256 is recorded. It is a per-user
temporal split, **not a globally leakage-free temporal evaluation**; other users'
training data can be later in time than a particular test event.

The local execution environment could not resolve the dataset host, so this path
was **not run against the real MovieLens file**. No real-data numbers are fabricated
or substituted. The downloaded dataset is not included in this repository.

## Peak memory: a negative result

Fresh Linux processes, large sparse case, rank 64, q=2, one worker:

| Model | Total process peak RSS |
|---|---:|
| sklearn | 250.859375 MiB |
| MyMF | 265.562500 MiB |

MyMF uses about **5.86% more** peak RSS here. This is `ru_maxrss` for the entire
process including imports, input and construction peaks, not an isolated algorithm
allocation measurement. The retained CSR transpose and float64 safety/statistics
work can increase peak memory. Avoiding a full dense reconstruction does not imply
using less memory than sklearn.

## Reproduce

```bash
python benchmarks/run.py --case sparse_medium --threads 1 --repeat 5 \
  --output benchmarks/results/local_medium_t1.json
python benchmarks/run.py --case sparse_medium --threads 4 --repeat 5 \
  --output benchmarks/results/local_medium_t4.json
python benchmarks/run.py --case sparse_large --threads 1 --repeat 3 \
  --output benchmarks/results/local_large_t1.json
python benchmarks/run.py --case sparse_large --threads 4 --repeat 3 \
  --output benchmarks/results/local_large_t4.json
python benchmarks/run.py --case dense --threads 1 --repeat 5 \
  --output benchmarks/results/local_dense.json
python benchmarks/recommendation.py --threads 1 --repeat 3 \
  --output benchmarks/results/local_recommendation.json
python benchmarks/memory.py --model mymf > benchmarks/results/local_memory_mymf.json
python benchmarks/memory.py --model sklearn > benchmarks/results/local_memory_sklearn.json
```

The environment, compiler, clock state, background load, cache behaviour, row
skew, dtype, spectrum and rank can all change results. FMA and different BLAS can
change the last numerical digits. Try an AVX2 override if AVX-512 is slower on your
processor. Changing BLAS thread budgets is a separate experiment, not a hidden
baseline adjustment.

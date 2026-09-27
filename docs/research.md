# Research selection and scope

Research reviewed for this release on 2026-09-27. "SOTA" is conditional on matrix
shape, sparsity, spectrum, rank, accuracy target, hardware and memory budget.
Published results from other hardware are not mymf benchmark results.

| Method | Primary source | Relevance and selection |
|---|---|---|
| Randomized SVD with power iteration | [Halko, Martinsson, Tropp (2011)](https://arxiv.org/abs/0909.4061); [sklearn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) | Widely used baseline; the installed sklearn 1.8.0 implementation is timed. |
| frPCA | [Fast Randomized PCA for Sparse Data (2018)](https://arxiv.org/abs/1810.06825) | Relevant sparse randomized-SVD implementation optimizations. Reviewed, not separately timed. |
| dashSVD | [Algorithm 1043, ACM TOMS (2024)](https://doi.org/10.1145/3660629); [paper](https://arxiv.org/abs/2404.09276); [authors' code](https://github.com/THU-numbda/dashSVD) | Selected core: dynamically shifted iteration aimed at fast low/moderate-accuracy sparse SVD. Independent implementation, not an upstream binary benchmark. |
| Randomized block Krylov | [Musco and Musco (2015)](https://arxiv.org/abs/1504.05477) | Accuracy/convergence alternative with an expanding basis. Not implemented or timed. |
| PRIMME_SVDS | [Authors' project](https://github.com/primme/primme) | Important accuracy-controlled solver; not timed, so superiority is not claimed. |
| PCAone / windowed randomized SVD | [Authors' project](https://github.com/Zilong-Li/PCAone) | Relevant large-data alternative; its broader data-handling objectives differ from this in-memory sklearn-style API. Not timed. |
| Row-aware Randomized SVD | [Paper](https://arxiv.org/abs/2408.04503); [2026 journal publication](https://doi.org/10.1002/nla.70088) | A newer row-aware range approximation was reviewed. Newer publication date alone does not establish lower runtime for this workload. Not implemented or timed. |

## What is and is not established

The release establishes that this implementation builds, has a tested Python API,
and is faster than the tested sklearn configuration on the supplied synthetic
sparse workloads at comparable measured reconstruction quality. It includes a
shift-off ablation, equal-iteration comparisons, reduced-iteration comparisons,
dense data, synthetic ranking metrics and a negative peak-memory result.

It does **not** establish a world speed record, superiority to the authors' MKL
implementation, equivalence at a specified certified residual, superiority to ALS
or BPR on recommendation, or generalization to real production data. The optional
MovieLens runner has no recorded real-data measurements in this release.

The scope is approximate uncentred SVD, not all algorithms called "matrix
factorization". Adding a masked or weighted recommendation objective would be a
different estimator, not an honest drop-in reinterpretation of TruncatedSVD.

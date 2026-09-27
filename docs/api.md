# API reference

```python
MyMF(n_components=2, *, algorithm="shifted", n_iter=2, n_oversamples=10,
     random_state=None, tol=0.0, normalizer="auto", orientation="auto",
     dtype="auto", n_jobs=None, blas_threads=1)
```

## Parameters

| Parameter | Meaning |
|---|---|
| `n_components` | Positive rank, no greater than `min(X.shape)`. |
| `algorithm` | `"shifted"` uses dynamic shifts. `"randomized"` disables shifts in the same implementation; it is not sklearn's complete solver. |
| `n_iter` | Nonnegative number of iterations; upper limit when `tol > 0`. Default 2. Shift first influences iteration 2. |
| `n_oversamples` | Nonnegative extra sketch width; actual width is capped at `min(X.shape)`. |
| `random_state` | Integer, NumPy `RandomState`, or None, following sklearn semantics. `Generator` is not accepted. |
| `tol` | Finite nonnegative heuristic stopping threshold. Zero disables stopping. Positive values need at least one oversampling direction. Not a certified error tolerance. |
| `normalizer` | `"auto"`: guarded float64 Gram eigendecomposition. `"qr"`: Householder QR plus small SVD. |
| `orientation` | `"auto"` maintains the basis on the smaller side; or force `"left"`/`"right"`. |
| `dtype` | `"auto"` preserves float32/float64 and converts integers to float64. Or `"float32"`/`"float64"`. |
| `n_jobs` | OpenMP sparse threads. None: min(8, quota-aware CPU count); -1: all quota CPUs; other negative values follow joblib-style counting. |
| `blas_threads` | Positive integer limit inside calls; None preserves ambient limits. Default 1. |

Nonfinite, complex, empty and incompatible-shape inputs are rejected. Dtypes are
not silently downcast to float32 in `"auto"`. Sparse inputs are converted to canonical
CSR without changing the caller's arrays. CSC, COO and SciPy sparse arrays work too.
Raw CSR indices are validated before native unchecked access. Public `_kernels`
functions are private implementation details and must not be fed malformed buffers.

## Methods

`fit(X, y=None)` fits and returns self. The ignored `y` supports sklearn pipelines.
It computes a training embedding internally because variance attributes need it;
there is no cheaper components-only fit path in this release.

`fit_transform(X, y=None)` returns a dense `(n_samples, n_components)` projection.
`transform(X_new)` projects rows with the **same item/feature columns**; it does
not support adding unseen feature columns without refitting. Transform inputs are
converted to the learned working dtype. Empty batches are rejected.

`inverse_transform(Z)` returns dense `(batch, n_features)` reconstruction. Do not
reconstruct the whole training matrix if it will not fit in memory.

`recommend(X_users, n_recommendations=10, *, filter_seen=True, batch_size=256,
item_batch_size=4096)` returns `(ids, scores)` arrays with shape `(users, count)`.
Item IDs are zero-based column indices. Any nonzero input entry is excluded when
`filter_seen=True`; its numerical sign does not change that rule. It ranks factor
dot products over **all** eligible columns using user and item blocks. There is
no approximate nearest-neighbour index. Too few eligible items produce `-1` IDs
and `-inf` scores. `n_recommendations` must not exceed the number of item columns.
Final results are sorted by descending score, with item-ID tie-breaking among
selected candidates; ties at the selection cutoff may select any equal-score item.
An all-zero cold-start user has no personalized signal; tied recommendations are
not a meaningful fallback policy.

`get_params`, `set_params`, `get_feature_names_out`, sklearn cloning, Pipeline and
`set_output(transform="pandas")` are inherited from sklearn. Pandas is optional;
it is needed only for pandas input/output. Feature column order is validated.

## Attributes after fitting

| Attribute | Shape or interpretation |
|---|---|
| `components_` | `(k, n_features)` approximately orthonormal right singular directions. |
| `singular_values_` | `(k,)` descending singular values of the compressed problem. |
| `explained_variance_` | Population variance of each training projection dimension, accumulated in float64. |
| `explained_variance_ratio_` | Explained variance divided by total feature variance, including sparse implicit zeros. A zero-variance matrix produces zeros. |
| `n_features_in_`, `feature_names_in_` | sklearn metadata; names exist when available on input. |
| `n_iter_`, `matrix_passes_` | Actual iterations and matrix products; passes exclude validation/statistics/transpose scans. |
| `n_fallbacks_` | Number of QR-path decompositions, including forced QR. |
| `orientation_`, `n_threads_` | Resolved basis side and requested sparse worker count. Serial builds still execute on one thread. |
| `shift_history_` | Applied shifts in internally scaled units, not original squared singular-value units. |
| `scale_` | Numerical rescaling factor; public factors remain in original units. |
| `pve_change_`, `stopped_early_` | Last heuristic and whether it stopped before the maximum iteration count. |

The compressed singular values need not exactly equal the column norms of the
public projection, especially with the left orientation. Explained variance and
uncentred projection energy are **different metrics**. Components can differ in
sign, rotation within repeated singular subspaces, or rounding between machines.

## Thread and persistence notes

`threadpoolctl` sets process-wide BLAS limits temporarily. For concurrent independent
calls from Python threads, configure external BLAS limits and use
`blas_threads=None`; otherwise thread-limit contexts can interfere. `fit` on the
same estimator instance is not thread-safe. Avoid CPU oversubscription when an
outer joblib loop also runs in parallel.

Pickle round trips are tested. Only load trusted pickle files; persistence is not
guaranteed across incompatible sklearn/NumPy/mymf versions.

## Differences from TruncatedSVD

No `algorithm="arpack"`, `power_iteration_normalizer` parameter or exact bitwise
parity is provided. MyMF defaults to two shifted iterations, not sklearn's five
randomized iterations. Its `tol` is a different, explicitly heuristic quantity.
Rank is bounded by both input dimensions. It accepts one-feature matrices. Input
is not centred. There is no `partial_fit`, GPU, missing-data mask, ALS, BPR, NMF,
regularization parameter, automatic rank selection or out-of-core operation.

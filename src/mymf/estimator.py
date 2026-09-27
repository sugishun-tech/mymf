"""scikit-learn-style non-centred matrix factorization."""
from __future__ import annotations

from contextlib import nullcontext
from numbers import Integral, Real

import numpy as np
from joblib import cpu_count
from scipy import sparse
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from ._threading import blas_context

from . import _kernels
from ._linalg import thin_svd


def _positive_int(name, value, *, zero=False):
    minimum = 0 if zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _threads(n_jobs):
    if n_jobs is None:
        return min(8, cpu_count())
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be None or a nonzero integer")
    # Explicit positive counts do not need affinity/cgroup discovery.
    return max(1, cpu_count() + 1 + n_jobs) if n_jobs < 0 else int(n_jobs)


def _canonical_csr(x):
    """Check raw CSR pointers before entering unchecked native code; never mutate x."""
    # A new container prevents check_format from pruning the caller's object.
    y = sparse.csr_matrix((x.data, x.indices, x.indptr), shape=x.shape, copy=False)
    y.check_format(full_check=True)
    if not y.has_canonical_format:
        y = y.copy()
        y.sum_duplicates()
        y.sort_indices()
    y.data = np.ascontiguousarray(y.data)
    y.indices = np.ascontiguousarray(y.indices)
    y.indptr = np.ascontiguousarray(y.indptr, dtype=y.indices.dtype)
    return y


def _matmat(x, b, n_threads):
    b = np.ascontiguousarray(b, dtype=x.dtype)
    if sparse.issparse(x):
        result = np.empty((x.shape[0], b.shape[1]), dtype=x.dtype)
        _kernels.csr_matmat(x.data, x.indices, x.indptr, b, result, n_threads)
        return result
    return np.ascontiguousarray(x @ b)


def _gaussian_sketch(random_state, shape, dtype):
    """Gaussian entries without generating temporary float64 data for float32.

    Integer seeds use PCG64 (v0.2+); explicit RandomState objects keep the legacy
    stream. None continues to respect numpy.random.seed, as in scikit-learn.
    """
    if isinstance(random_state, Integral) and not isinstance(random_state, (bool, np.bool_)):
        rng = np.random.default_rng(int(random_state))
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = check_random_state(random_state)
    if isinstance(rng, np.random.Generator):
        return rng.standard_normal(shape, dtype=dtype)
    return np.ascontiguousarray(rng.standard_normal(shape), dtype=dtype)


class MyMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Fast approximate truncated SVD, without centering or missing-value masking.

    Parameters
    ----------
    n_components : int, default=2
        Output rank, at most min(n_samples, n_features).
    algorithm : {"shifted", "randomized"}, default="shifted"
        Dynamic-shift subspace iteration (dashSVD Algorithm 4), or the same
        implementation with shifts disabled for a controlled ablation.
    n_iter : int, default=2
        Number of power iterations, or maximum if tol > 0. Zero is valid.
        A shift is first estimated after iteration 1 and used at iteration 2.
    n_oversamples : int, default=10
        Additional sketch columns. The sketch is capped at min(X.shape).
    random_state : int, Generator, RandomState or None, default=None
        Gaussian sketch seed. Integers use NumPy PCG64 and direct float32 draws
        in v0.2+, so their factors differ from v0.1. Explicit RandomState objects
        retain the legacy stream. None respects numpy.random.seed.
    tol : float, default=0
        Zero disables early stopping. A positive value enables a heuristic
        PVE-surrogate change test; it is NOT a certified SVD error tolerance.
        Requires n_components + 1 <= min(X.shape) and n_oversamples >= 1.
    normalizer : {"auto", "qr"}, default="auto"
        Float64 Gram eigendecomposition with conditioning guards, or always
        Householder QR + small SVD. QR is safer for ill-conditioned inputs.
    orientation : {"auto", "left", "right"}, default="auto"
        The side on which to maintain the iterative basis. Auto uses the
        smaller dimension. Right avoids an extra training projection pass.
    dtype : {"auto", "float32", "float64"}, default="auto"
        Auto preserves float32/float64 input; converts integer data to float64.
        Float32 reduces sparse bandwidth and storage, at lower precision.
    n_jobs : int or None, default=None
        Sparse OpenMP threads. None uses min(8, CPU quota); -1 uses the quota.
        Does not control BLAS. Without OpenMP kernels run serially.
    blas_threads : int or None, default=1
        BLAS thread limit inside calls. None leaves the ambient setting alone.
        Threadpoolctl changes process-wide limits, so concurrent model calls
        should instead set external limits and use blas_threads=None.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Approximately orthonormal right singular vectors.
    singular_values_ : ndarray of shape (n_components,)
        Singular values of the compressed problem, in descending order.
    explained_variance_, explained_variance_ratio_ : ndarray
        Population variance of X @ components_.T and its fraction of total
        feature variance. Constant input has a zero variance ratio.
    n_iter_ : int
        Actual completed power iterations.
    matrix_passes_ : int
        Full-width matrix products during fitting; does not include validation,
        transpose construction, or the linear scans for variance statistics.
    n_fallbacks_ : int
        Decompositions that used QR instead of the guarded Gram path.
    shift_history_ : list[float]
        Shift applied on each iteration, in internally scaled matrix units.
    pve_change_ : float or None
        Last heuristic stopping statistic; no certified error interpretation.
    stopped_early_ : bool
        Whether the heuristic stopped iterations before their configured maximum.
    orientation_ : str
        Resolved basis side.
    scale_ : float
        Input scaling for unusually large/small values; the public factors are
        always returned in the original units.

    Notes
    -----
    This minimizes an unweighted zero-filled reconstruction objective. It is
    NOT implicit ALS, observed-only rating factorization, NMF, or BPR.
    """

    def __init__(self, n_components=2, *, algorithm="shifted", n_iter=2,
                 n_oversamples=10, random_state=None, tol=0.0, normalizer="auto",
                 orientation="auto", dtype="auto", n_jobs=None, blas_threads=1):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.tol = tol
        self.normalizer = normalizer
        self.orientation = orientation
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.blas_threads = blas_threads

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float32", "float64"]
        return tags

    def _parameters(self):
        _positive_int("n_components", self.n_components)
        _positive_int("n_iter", self.n_iter, zero=True)
        _positive_int("n_oversamples", self.n_oversamples, zero=True)
        if self.algorithm not in ("shifted", "randomized"):
            raise ValueError("algorithm must be 'shifted' or 'randomized'")
        if self.normalizer not in ("auto", "qr"):
            raise ValueError("normalizer must be 'auto' or 'qr'")
        if self.orientation not in ("auto", "left", "right"):
            raise ValueError("orientation must be 'auto', 'left', or 'right'")
        if self.dtype not in ("auto", "float32", "float64"):
            raise ValueError("dtype must be 'auto', 'float32', or 'float64'")
        if isinstance(self.tol, (bool, np.bool_)) or not isinstance(self.tol, Real):
            raise ValueError("tol must be a finite, nonnegative number")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a finite, nonnegative number")
        if self.blas_threads is not None:
            _positive_int("blas_threads", self.blas_threads)
        return _threads(self.n_jobs)

    def _context(self):
        return (nullcontext() if self.blas_threads is None else
                blas_context(int(self.blas_threads)))

    def _input(self, x, *, reset):
        if reset:
            target_dtype = [np.float64, np.float32] if self.dtype == "auto" else self.dtype
        else:
            target_dtype = self.components_.dtype
        x = validate_data(self, x, reset=reset, accept_sparse="csr",
                          dtype=target_dtype, order=None, copy=False,
                          ensure_min_samples=1, ensure_min_features=1)
        return _canonical_csr(x) if sparse.issparse(x) else x

    def fit(self, X, y=None):
        """Fit the factorization. y is ignored; return self."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit and return the dense embedding X @ components_.T."""
        n_threads = self._parameters()
        x = self._input(X, reset=True)
        m, n = x.shape
        k = int(self.n_components)
        if k > min(m, n):
            raise ValueError("n_components must be <= min(n_samples, n_features)")
        width = min(k + int(self.n_oversamples), m, n)
        if self.tol > 0 and width <= k:
            raise ValueError("tol > 0 requires at least one oversampling direction")
        self.n_iter_ = 0
        self.matrix_passes_ = 0
        self.n_fallbacks_ = 0
        self.shift_history_ = []
        self.pve_change_ = None
        self.stopped_early_ = False
        self.orientation_ = self.orientation
        if self.orientation_ == "auto":
            self.orientation_ = "right" if n <= m else "left"
        self._n_features_out = k
        self.n_threads_ = n_threads

        with self._context():
            # Rescale only out-of-normal-range inputs to avoid overflow during powers.
            data = x.data if sparse.issparse(x) else x
            maxabs = float(np.max(np.abs(data), initial=0.0))
            self.scale_ = maxabs if maxabs and (maxabs < 1e-3 or maxabs > 1e3) else 1.0
            a = x / self.scale_ if self.scale_ != 1.0 else x
            if sparse.issparse(a):
                # _input validated and canonicalized x. Scalar scaling and
                # SciPy's transpose preserve that invariant; do not scan again.
                at = a.T.tocsr()
            else:
                at = a.T
            # Iteration lives in the smaller side unless the user overrides it.
            forward, backward = (a, at) if self.orientation_ == "right" else (at, a)
            omega = _gaussian_sketch(
                self.random_state, (forward.shape[0], width), a.dtype
            )
            q = _matmat(backward, omega, n_threads)
            self.matrix_passes_ += 1
            del omega
            q, _, _, fallback = thin_svd(q, normalizer=self.normalizer)
            self.n_fallbacks_ += int(fallback)
            alpha = 0.0
            previous = None
            for iteration in range(int(self.n_iter)):
                b = _matmat(forward, q, n_threads)
                next_q = _matmat(backward, b, n_threads)
                del b
                self.matrix_passes_ += 2
                self.shift_history_.append(alpha)
                if alpha:
                    _kernels.subtract_shift(next_q, q, a.dtype.type(alpha), n_threads)
                q, shifted_s, _, fallback = thin_svd(next_q, normalizer=self.normalizer)
                del next_q
                self.n_fallbacks_ += int(fallback)
                self.n_iter_ = iteration + 1
                surrogate = shifted_s.astype(np.float64) + alpha
                if self.tol > 0 and previous is not None:
                    denominator = max(float(surrogate[k]), np.finfo(np.float64).tiny)
                    self.pve_change_ = float(np.max(np.abs(surrogate[:k] - previous[:k]))
                                             / denominator)
                    if self.pve_change_ <= self.tol:
                        self.stopped_early_ = self.n_iter_ < self.n_iter
                        break
                # Store the surrogate with the shift actually used, not the next shift.
                previous = surrogate
                if self.algorithm == "shifted" and float(shifted_s[-1]) > alpha:
                    alpha = (float(shifted_s[-1]) + alpha) * 0.5

            b = _matmat(forward, q, n_threads)
            self.matrix_passes_ += 1
            left = self.orientation_ == "left"
            ub, s, vt, fallback = thin_svd(b, normalizer=self.normalizer, left=left)
            self.n_fallbacks_ += int(fallback)
            if left:
                self.components_ = np.ascontiguousarray(ub[:, :k].T)
                # Unlike U*S of the range approximation, this is the public projection.
                z = _matmat(a, self.components_.T, n_threads)
                self.matrix_passes_ += 1
            else:
                rotation = np.ascontiguousarray(vt[:k].T)
                self.components_ = np.ascontiguousarray((q @ rotation).T)
                # Reuse the last wide sketch; no extra sparse matrix pass.
                z = np.ascontiguousarray(b @ rotation)
            del b, q, a, at, forward, backward
            self.singular_values_ = np.ascontiguousarray(s[:k] * self.scale_)
            z *= self.scale_
            # Pin the sign of every component by its largest-magnitude loading.
            anchors = np.argmax(np.abs(self.components_), axis=1)
            signs = np.sign(self.components_[np.arange(k), anchors])
            signs[signs == 0] = 1
            self.components_ *= signs[:, None]
            z *= signs[None, :]
            if not np.isfinite(z).all() or not np.isfinite(self.singular_values_).all():
                raise FloatingPointError("Output overflows selected dtype; use dtype='float64'.")
            self.explained_variance_ = np.var(z, axis=0, dtype=np.float64)
            if sparse.issparse(x):
                # Columnwise Welford moments plus the implicit zeros. No
                # float64 copy of every sparse value is needed.
                total_variance = _kernels.csr_total_variance(
                    x.data, x.indices, m, np.zeros(n, dtype=np.float64),
                    np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int64)
                )
            else:
                total_variance = float(np.var(x, axis=0, dtype=np.float64).sum())
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance if total_variance > 0
                else np.zeros(k, dtype=np.float64)
            )
        return z

    def transform(self, X):
        """Project new rows to the learned feature space; never refit."""
        check_is_fitted(self, "components_")
        x = self._input(X, reset=False)
        with self._context():
            return _matmat(x, self.components_.T, _threads(self.n_jobs))

    def inverse_transform(self, X):
        """Reconstruct dense rows from embeddings (may allocate a large matrix)."""
        check_is_fitted(self, "components_")
        z = check_array(X, dtype=self.components_.dtype, ensure_min_features=1)
        if z.shape[1] != self.components_.shape[0]:
            raise ValueError("embedding width must equal n_components")
        with self._context():
            return z @ self.components_

    def recommend(self, X, n_recommendations=10, *, filter_seen=True,
                  batch_size=256, item_batch_size=4096):
        """Return (item_ids, scores) without a users-by-items score allocation.

        X has one row per requested user and the training feature layout. Any
        nonzero in X is excluded when filter_seen=True. Unavailable slots have
        id=-1 and score=-inf. Ties at the cutoff may choose any equal-score item.
        Scores are factorization dot products, not probabilities.
        """
        check_is_fitted(self, "components_")
        _positive_int("n_recommendations", n_recommendations)
        _positive_int("batch_size", batch_size)
        _positive_int("item_batch_size", item_batch_size)
        if not isinstance(filter_seen, (bool, np.bool_)):
            raise ValueError("filter_seen must be boolean")
        n_items = self.n_features_in_
        if n_recommendations > n_items:
            raise ValueError("n_recommendations must not exceed n_features")
        x = self._input(X, reset=False)
        count = int(n_recommendations)
        n_threads = _threads(self.n_jobs)
        all_ids = np.empty((x.shape[0], count), dtype=np.int64)
        all_scores = np.empty((x.shape[0], count), dtype=self.components_.dtype)
        with self._context():
            for start in range(0, x.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), x.shape[0])
                rows = x[start:stop]
                z = _matmat(rows, self.components_.T, n_threads)
                best_scores = np.full((stop - start, count), -np.inf, dtype=z.dtype)
                best_ids = np.full((stop - start, count), -1, dtype=np.int64)
                for lo in range(0, n_items, int(item_batch_size)):
                    hi = min(lo + int(item_batch_size), n_items)
                    scores = np.ascontiguousarray(z @ self.components_[:, lo:hi])
                    if filter_seen:
                        if sparse.issparse(rows):
                            _kernels.mask_seen(rows.data, rows.indices, rows.indptr,
                                               scores, lo, n_threads)
                        else:
                            scores[rows[:, lo:hi] != 0] = -np.inf
                    scores = np.concatenate((best_scores, scores), axis=1)
                    ids = np.concatenate((best_ids, np.broadcast_to(
                        np.arange(lo, hi, dtype=np.int64), (stop - start, hi - lo))), axis=1)
                    take = np.argpartition(scores, scores.shape[1] - count, axis=1)[:, -count:]
                    best_scores = np.take_along_axis(scores, take, axis=1)
                    best_ids = np.take_along_axis(ids, take, axis=1)
                order = np.lexsort((best_ids, -best_scores), axis=1)
                best_scores = np.take_along_axis(best_scores, order, axis=1)
                best_ids = np.take_along_axis(best_ids, order, axis=1)
                best_ids[~np.isfinite(best_scores)] = -1
                all_ids[start:stop], all_scores[start:stop] = best_ids, best_scores
        return all_ids, all_scores

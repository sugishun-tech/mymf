import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from mymf import MyMF, build_info
from mymf import _kernels
from mymf._linalg import thin_svd


@pytest.mark.parametrize('shape', [(100, 60), (60, 100)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('form', ['dense', 'csr', 'csc', 'coo', 'array'])
@pytest.mark.parametrize('side', ['auto', 'left', 'right'])
def test_projection(shape, dtype, form, side):
    x = np.random.RandomState(3).normal(size=shape).astype(dtype)
    if form != 'dense':
        constructor = {'csr': sparse.csr_matrix, 'csc': sparse.csc_matrix,
                       'coo': sparse.coo_matrix, 'array': sparse.csr_array}[form]
        x = constructor(x)
    model = MyMF(10, random_state=42, orientation=side, n_jobs=2)
    z = model.fit_transform(x)
    atol = 3e-5 if dtype == np.float32 else 1e-9
    assert z.dtype == dtype
    assert z.shape == (shape[0], 10)
    assert_allclose(z, model.transform(x), atol=atol)
    assert_allclose(model.components_ @ model.components_.T, np.eye(10), atol=atol)
    assert np.all(np.diff(model.singular_values_) <= 0)
    assert_allclose(model.inverse_transform(z), z @ model.components_)
    assert model.n_features_in_ == shape[1]
    assert model.matrix_passes_ == 6 + (model.orientation_ == 'left')


@pytest.mark.parametrize('shape', [(30, 20), (20, 30), (1, 4), (4, 1)])
@pytest.mark.parametrize('value', [0., 1.])
@pytest.mark.parametrize('sp', [False, True])
def test_degenerate(shape, value, sp):
    x = np.full(shape, value)
    if sp:
        x = sparse.csr_matrix(x)
    model = MyMF(min(4, *shape), random_state=5)
    z = model.fit_transform(x)
    assert np.isfinite(z).all()
    assert np.isfinite(model.explained_variance_ratio_).all()
    assert_allclose(z, model.transform(x), atol=1e-10)
    assert_allclose(model.inverse_transform(z), x.toarray() if sp else x, atol=1e-10)
    assert_allclose(model.explained_variance_ratio_, 0)


@pytest.mark.parametrize('normalizer', ['auto', 'qr'])
@pytest.mark.parametrize('shape', [(70, 45), (45, 70)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_low_rank(normalizer, shape, dtype):
    rng = np.random.RandomState(7)
    x = (rng.normal(size=(shape[0], 5)) @ rng.normal(size=(5, shape[1]))).astype(dtype)
    model = MyMF(5, n_iter=3, random_state=3, normalizer=normalizer)
    z = model.fit_transform(x)
    assert_allclose(z @ model.components_, x, atol=2e-4 if dtype == np.float32 else 2e-10)
    assert model.n_fallbacks_ > 0


@pytest.mark.parametrize('shape', [(50, 30), (30, 50)])
def test_full_rank(shape):
    x = np.random.RandomState(1).normal(size=shape)
    model = MyMF(min(shape), random_state=0)
    z = model.fit_transform(x)
    assert_allclose(z @ model.components_, x, atol=1e-10)
    assert_allclose(model.singular_values_, np.linalg.svd(x, compute_uv=False), atol=1e-10)


@pytest.mark.parametrize('scale', [1e-100, 1e-12, 1e12, 1e100])
def test_extreme_finite_scale(scale):
    r = np.random.RandomState(5)
    x = r.normal(size=(30, 20)) * scale
    model = MyMF(20, random_state=1)
    z = model.fit_transform(x)
    assert_allclose((z @ model.components_) / scale, x / scale, atol=1e-8)


def test_variance_matches_definition():
    x = sparse.random(100, 60, density=.3, random_state=0, format='csr')
    model = MyMF(10, random_state=1)
    z = model.fit_transform(x)
    assert_allclose(model.explained_variance_, z.var(axis=0), atol=1e-12)
    assert_allclose(model.explained_variance_ratio_, z.var(axis=0) / x.toarray().var(axis=0).sum())


def test_duplicate_unsorted_and_no_mutation():
    x = sparse.csr_matrix((np.array([1., 2., 3., -3., 4.]),
                           np.array([3, 1, 2, 2, 1]), np.array([0, 3, 5, 5])), shape=(3, 5))
    old = [x.data.copy(), x.indices.copy(), x.indptr.copy()]
    model = MyMF(2, random_state=1)
    z = model.fit_transform(x)
    assert_allclose(z, model.transform(x.toarray()), atol=1e-10)
    for before, after in zip(old, [x.data, x.indices, x.indptr]):
        assert np.array_equal(before, after)


@pytest.mark.parametrize('idx_dtype', [np.int32, np.int64])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('threads', [1, 4])
def test_native_kernel(idx_dtype, dtype, threads):
    x = sparse.random(1200, 200, density=.1, random_state=6, format='csr', dtype=dtype)
    b = np.ascontiguousarray(np.random.RandomState(3).normal(size=(200, 40)), dtype=dtype)
    out = np.empty((1200, 40), dtype=dtype)
    indices = x.indices.astype(idx_dtype)
    indptr = x.indptr.astype(idx_dtype)
    for arr in (x.data, b, indices, indptr):
        arr.flags.writeable = False
    _kernels.csr_matmat(x.data, indices, indptr, b, out, threads)
    assert_allclose(out, x @ b, atol=1e-5 if dtype == np.float32 else 1e-12)


def test_readonly_model_input():
    x = sparse.random(60, 40, density=.5, random_state=1, format='csr')
    for arr in (x.data, x.indices, x.indptr):
        arr.flags.writeable = False
    model = MyMF(5, random_state=2)
    assert np.isfinite(model.fit_transform(x)).all()


def test_dense_noncontiguous():
    x = np.random.RandomState(5).normal(size=(100, 120))[::2, ::3]
    model = MyMF(6, random_state=2)
    assert_allclose(model.fit_transform(x), model.transform(x), atol=1e-10)


def test_reproducibility_pickle_clone_pipeline():
    x = sparse.random(80, 50, density=.3, random_state=3, format='csr')
    first = MyMF(8, random_state=4, n_jobs=1)
    second = clone(first).set_params(n_jobs=3)
    first.fit(x)
    second.fit(x)
    assert_allclose(first.components_, second.components_, atol=1e-10)
    recovered = pickle.loads(pickle.dumps(first))
    assert_allclose(first.transform(x), recovered.transform(x))
    pipeline = make_pipeline(clone(first), Normalizer())
    assert pipeline.fit_transform(x).shape == (80, 8)
    assert list(first.get_feature_names_out()) == [f'mymf{i}' for i in range(8)]


@pytest.mark.parametrize('key,value', [
    ('n_components', 0), ('n_components', True), ('n_components', 2.3),
    ('n_iter', -1), ('n_iter', False), ('n_oversamples', -1),
    ('algorithm', 'made_up'), ('normalizer', 'bad'), ('orientation', 'bad'),
    ('dtype', 'float16'), ('n_jobs', 0), ('n_jobs', True),
    ('blas_threads', 0), ('tol', -1), ('tol', np.inf), ('tol', np.nan), ('tol', True)
])
def test_invalid_parameters(key, value):
    with pytest.raises(ValueError):
        MyMF(**{key: value}).fit(np.eye(8))


@pytest.mark.parametrize('x', [np.array([[np.nan, 1], [1, 1]]),
                              np.array([[np.inf, 1], [1, 1]]),
                              np.ones((2, 2), dtype=complex),
                              np.zeros((0, 5)), np.zeros((5, 0)), np.ones(4)])
def test_invalid_input(x):
    with pytest.raises(ValueError):
        MyMF().fit(x)


def test_invalid_sparse_structure():
    x = sparse.eye(10, format='csr')
    x.indices[0] = 99999
    with pytest.raises(ValueError):
        MyMF().fit(x)


def test_dimension_and_fitted_checks():
    with pytest.raises(NotFittedError):
        MyMF().transform(np.ones((2, 2)))
    with pytest.raises(ValueError):
        MyMF(10).fit(np.ones((3, 8)))
    with pytest.raises(ValueError):
        MyMF(3, n_oversamples=0, tol=.1).fit(np.eye(6))
    model = MyMF(2).fit(np.eye(5))
    with pytest.raises(ValueError):
        model.transform(np.ones((3, 6)))
    with pytest.raises(ValueError):
        model.inverse_transform(np.ones((3, 6)))


def test_early_stop_and_disabled_stop():
    x = np.random.RandomState(1).normal(size=(60, 40))
    model = MyMF(5, n_iter=10, tol=1e10, random_state=1).fit(x)
    assert model.n_iter_ == 2
    assert model.stopped_early_
    assert model.pve_change_ is not None
    fixed = MyMF(5, n_iter=4, tol=0, random_state=1).fit(x)
    assert fixed.n_iter_ == 4
    assert not fixed.stopped_early_
    assert fixed.shift_history_[0] == 0
    assert fixed.shift_history_[1] > 0
    plain = MyMF(5, n_iter=4, algorithm='randomized', random_state=1).fit(x)
    assert plain.shift_history_ == [0, 0, 0, 0]


def test_shift_improves_controlled_spectrum():
    rng = np.random.RandomState(0)
    u = np.linalg.qr(rng.normal(size=(150, 80)))[0]
    v = np.linalg.qr(rng.normal(size=(100, 80)))[0]
    x = (u * np.linspace(10, 1, 80)) @ v.T
    shifted = MyMF(10, n_iter=4, random_state=10).fit_transform(x)
    ordinary = MyMF(10, n_iter=4, random_state=10, algorithm='randomized').fit_transform(x)
    assert np.linalg.norm(shifted) > np.linalg.norm(ordinary)


@pytest.mark.parametrize('sp', [True, False])
def test_recommend_matches_dense_scores(sp):
    x = sparse.random(70, 55, density=.1, random_state=0, format='csr')
    if not sp:
        x = x.toarray()
    model = MyMF(12, random_state=0).fit(x)
    query = x[:8]
    ids, scores = model.recommend(query, 10, batch_size=3, item_batch_size=7)
    dense_scores = model.transform(query) @ model.components_
    dense_scores[(query.toarray() if sp else query) != 0] = -np.inf
    expected = np.argsort(-dense_scores, axis=1)[:, :10]
    assert np.array_equal(ids, expected)
    assert_allclose(scores, np.take_along_axis(dense_scores, expected, axis=1), atol=1e-10)


def test_recommend_all_seen_and_invalid():
    model = MyMF(2, random_state=2).fit(np.ones((10, 8)))
    ids, scores = model.recommend(np.ones((1, 8)), 5, item_batch_size=2)
    assert np.all(ids == -1)
    assert np.all(scores == -np.inf)
    for options in [{'n_recommendations': 0}, {'n_recommendations': 9},
                    {'batch_size': 0}, {'item_batch_size': 0}, {'filter_seen': 'no'}]:
        with pytest.raises(ValueError):
            model.recommend(np.ones((1, 8)), **options)


def test_thin_svd_fallback():
    x = np.diag(np.geomspace(1, 1e-14, 8))
    u, s, vt, fallback = thin_svd(x)
    assert fallback
    assert_allclose(u @ np.diag(s) @ vt, x, atol=1e-15)
    assert_allclose(u.T @ u, np.eye(8), atol=1e-12)


def test_compiled_extension_is_real():
    assert _kernels.__file__.endswith(('.so', '.pyd'))
    assert build_info()['language'] == 'C'

@pytest.mark.parametrize('shape,side', [((83, 47), 'right'), ((47, 83), 'left')])
@pytest.mark.parametrize('normalizer', ['auto', 'qr'])
def test_against_independent_shifted_svd_reference(shape, side, normalizer):
    """Use full thin LAPACK SVD at every step, not the production Gram helper."""
    from scipy import linalg
    x = np.random.RandomState(91).normal(size=shape)
    k, width, iterations, seed = 7, 11, 3, 17
    forward, backward = (x, x.T) if side == 'right' else (x.T, x)
    omega = np.random.RandomState(seed).normal(size=(forward.shape[0], width))
    q = linalg.svd(backward @ omega, full_matrices=False)[0]
    alpha = 0.0
    history = []
    for _ in range(iterations):
        history.append(alpha)
        q, s, _ = linalg.svd(backward @ (forward @ q) - alpha * q,
                             full_matrices=False)
        if s[-1] > alpha:
            alpha = (s[-1] + alpha) / 2
    b = forward @ q
    u, s, vt = linalg.svd(b, full_matrices=False)
    expected = u[:, :k].T if side == 'left' else (q @ vt[:k].T).T
    model = MyMF(k, n_oversamples=width-k, n_iter=iterations,
                 random_state=seed, orientation=side, normalizer=normalizer, n_jobs=1)
    z = model.fit_transform(x)
    # Compare projectors, which are invariant to arbitrary SVD signs.
    assert_allclose(model.components_.T @ model.components_, expected.T @ expected,
                    atol=2e-10)
    assert_allclose(model.singular_values_, s[:k], rtol=2e-10)
    assert_allclose(model.shift_history_, history, rtol=2e-10)
    assert_allclose(z, x @ model.components_.T, atol=2e-10)


def test_pandas_feature_validation_and_output():
    pd = pytest.importorskip('pandas')
    x = pd.DataFrame(np.random.RandomState(2).normal(size=(30, 5)),
                     columns=['a', 'b', 'c', 'd', 'e'])
    model = MyMF(2, random_state=1).set_output(transform='pandas')
    z = model.fit_transform(x)
    assert isinstance(z, pd.DataFrame)
    assert list(z.columns) == ['mymf0', 'mymf1']
    assert_allclose(z, model.transform(x), atol=1e-10)
    with pytest.raises(ValueError, match='feature names'):
        model.transform(x[['b', 'a', 'c', 'd', 'e']])


def test_sklearn_estimator_checks():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(MyMF(n_jobs=1))

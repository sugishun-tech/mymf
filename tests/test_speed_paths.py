"""Numerical/safety regression tests for the v0.2 speed paths (not timing tests)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from mymf import MyMF, _kernels, refresh_threadpools
from mymf._threading import _get_controller


@pytest.mark.parametrize('width', [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                   40, 41, 42, 48, 63, 64, 65, 74, 96, 127, 128,
                                   129, 160])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_register_tails(width, dtype, index_dtype):
    rng = np.random.default_rng(812)
    # The final feature is used: masked loads must not overread the last row.
    a = sparse.random(23, 37, density=.22, format='csr', random_state=4,
                      dtype=dtype).tolil()
    a[0] = 0
    a[-1, -1] = -1
    a = a.tocsr()
    a.indices = a.indices.astype(index_dtype)
    a.indptr = a.indptr.astype(index_dtype)
    b = rng.standard_normal((37, width)).astype(dtype)
    result = np.empty((23, width), dtype=dtype)
    for x in (a.data, a.indices, a.indptr, b):
        x.flags.writeable = False
    _kernels.csr_matmat(a.data, a.indices, a.indptr, b, result, 1)
    assert_allclose(result, a @ b, rtol=2e-6 if dtype == np.float32 else 1e-13,
                    atol=2e-6 if dtype == np.float32 else 1e-13)
    assert not np.any(result[0])


@pytest.mark.parametrize('width', [1, 7, 31, 42, 64, 74, 128, 129])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_parallel_register_rows(width, dtype):
    a = sparse.random(500, 150, density=.2, format='csr', random_state=4, dtype=dtype)
    b = np.random.default_rng(11).normal(size=(150, width)).astype(dtype)
    serial = np.empty((500, width), dtype=dtype)
    parallel = np.empty_like(serial)
    _kernels.csr_matmat(a.data, a.indices, a.indptr, b, serial, 1)
    _kernels.csr_matmat(a.data, a.indices, a.indptr, b, parallel, 3)
    assert_allclose(serial, parallel, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('pattern', ['zero', 'one', 'binary', 'signed', 'stored_zero',
                                     'large_constant', 'near_constant'])
def test_native_variance(dtype, pattern):
    rng = np.random.default_rng(91)
    dense = rng.standard_normal((87, 19)).astype(dtype)
    if pattern == 'zero':
        dense[:] = 0
    elif pattern == 'one':
        dense[:] = 1
    elif pattern == 'binary':
        dense = (dense > .7).astype(dtype)
    elif pattern == 'large_constant':
        dense[:] = 1e12
    elif pattern == 'near_constant':
        dense = (dense.astype(np.float64) + 1e6).astype(dtype)
    else:
        dense[dense < -.3] = 0
    a = sparse.csr_matrix(dense)
    if pattern == 'stored_zero':
        a.data[::3] = 0
        dense = a.toarray()
    for index_dtype in (np.int32, np.int64):
        got = _kernels.csr_total_variance(a.data, a.indices.astype(index_dtype),
                                        a.shape[0], np.zeros(a.shape[1]),
                                        np.zeros(a.shape[1]),
                                        np.zeros(a.shape[1], dtype=np.int64))
        expected = dense.astype(np.float64).var(axis=0).sum()
        assert_allclose(got, expected, rtol=2e-10, atol=1e-12)


def test_thread_controller_reused_and_restored():
    refresh_threadpools()
    first = _get_controller()
    a = sparse.eye(30, format='csr')
    with threadpool_limits(2, user_api='blas'):
        model = MyMF(3, n_jobs=1, blas_threads=1, random_state=1).fit(a)
        model.transform(a)
        assert _get_controller() is first
        assert all(x['num_threads'] == 2 for x in threadpool_info()
                   if x['user_api'] == 'blas')
    refresh_threadpools()
    assert _get_controller() is not first


def test_positive_n_jobs_does_not_discover_cpu_quota(monkeypatch):
    import mymf.estimator as estimator
    def fail():
        raise AssertionError('CPU quota discovery is unnecessary for n_jobs=1')
    monkeypatch.setattr(estimator, 'cpu_count', fail)
    MyMF(2, n_jobs=1, random_state=1).fit(np.eye(4))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_generator_reproducible_and_none_respects_global_seed(dtype):
    a = sparse.random(100, 50, density=.2, format='csr', random_state=9, dtype=dtype)
    for state in ('integer', 'generator', 'legacy', 'none'):
        def fit():
            np.random.seed(123)
            s = {'integer': 123, 'generator': np.random.default_rng(123),
                 'legacy': np.random.RandomState(123), 'none': None}[state]
            return MyMF(8, random_state=s, n_jobs=1).fit(a).components_
        assert_allclose(fit(), fit(), rtol=0, atol=0)


def test_fresh_fit_reads_modified_input_no_factor_cache():
    a = sparse.random(90, 50, density=.25, format='csr', random_state=1, dtype=np.float64)
    first = MyMF(8, random_state=5, n_jobs=1).fit(a).components_
    a.data[::5] *= 9
    second = MyMF(8, random_state=5, n_jobs=1).fit(a).components_
    assert np.linalg.norm(first.T @ first - second.T @ second) > .1


@pytest.mark.parametrize('mutation', ['pointer_order', 'negative_index', 'large_index'])
def test_sparse_safety_checks_still_present(mutation):
    a = sparse.eye(30, format='csr')
    if mutation == 'pointer_order':
        a.indptr[5] = a.indptr[6] + 1
    elif mutation == 'negative_index':
        a.indices[0] = -1
    else:
        a.indices[0] = 30
    with pytest.raises(ValueError):
        MyMF(3, n_jobs=1).fit(a)

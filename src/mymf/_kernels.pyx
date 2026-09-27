# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True
"""Private kernels. CSR structure is validated at the estimator boundary."""
from cython.parallel cimport prange
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memset
from libc.math cimport INFINITY

ctypedef fused real_t:
    float
    double

ctypedef fused index_t:
    int32_t
    int64_t

cdef extern from "_simd.h" nogil:
    int mymf_openmp_enabled() noexcept
    int mymf_native_enabled() noexcept
    const char* mymf_simd_name() noexcept
    void mymf_init() noexcept
    void mymf_row_f32_i32(const float*, const int32_t*, size_t, size_t,
                          const float*, float*, size_t) noexcept
    void mymf_row_f32_i64(const float*, const int64_t*, size_t, size_t,
                          const float*, float*, size_t) noexcept
    void mymf_row_f64_i32(const double*, const int32_t*, size_t, size_t,
                          const double*, double*, size_t) noexcept
    void mymf_row_f64_i64(const double*, const int64_t*, size_t, size_t,
                          const double*, double*, size_t) noexcept

mymf_init()


def build_info():
    return {"openmp": bool(mymf_openmp_enabled()),
            "native": bool(mymf_native_enabled()), "language": "C",
            "simd": mymf_simd_name().decode("ascii")}


cdef inline void _row(
    const real_t[::1] data, const index_t[::1] indices,
    const index_t[::1] indptr, const real_t[:, ::1] b,
    real_t[:, ::1] out, Py_ssize_t i, Py_ssize_t width
) noexcept nogil:
    # Empty CSR buffers cannot be indexed, even for a zero-length loop.
    if indptr[i] == indptr[i + 1]:
        memset(&out[i, 0], 0, width * sizeof(real_t))
        return
    if real_t is float:
        if index_t is int32_t:
            mymf_row_f32_i32(&data[0], &indices[0], indptr[i], indptr[i + 1],
                             &b[0, 0], &out[i, 0], width)
        else:
            mymf_row_f32_i64(&data[0], &indices[0], indptr[i], indptr[i + 1],
                             &b[0, 0], &out[i, 0], width)
    else:
        if index_t is int32_t:
            mymf_row_f64_i32(&data[0], &indices[0], indptr[i], indptr[i + 1],
                             &b[0, 0], &out[i, 0], width)
        else:
            mymf_row_f64_i64(&data[0], &indices[0], indptr[i], indptr[i + 1],
                             &b[0, 0], &out[i, 0], width)


def csr_matmat(
    const real_t[::1] data, const index_t[::1] indices,
    const index_t[::1] indptr, const real_t[:, ::1] b,
    real_t[:, ::1] out, int n_threads=1
):
    """out = CSR @ b. Internal: indices must already be checked; no aliasing."""
    cdef Py_ssize_t m = indptr.shape[0] - 1
    cdef Py_ssize_t width = b.shape[1]
    cdef Py_ssize_t i
    if n_threads < 1 or m < 0 or width < 1:
        raise ValueError("invalid kernel dimensions or thread count")
    if out.shape[0] != m or out.shape[1] != width:
        raise ValueError("out shape mismatch")
    # Guided scheduling also handles popularity-skewed row lengths.
    for i in prange(m, nogil=True, num_threads=n_threads, schedule='guided',
                    use_threads_if=(data.shape[0] * width >= 200000)):
        _row(data, indices, indptr, b, out, i, width)


def subtract_shift(real_t[:, ::1] y, const real_t[:, ::1] q,
                   real_t alpha, int n_threads=1):
    cdef Py_ssize_t i, j
    if y.shape[0] != q.shape[0] or y.shape[1] != q.shape[1] or n_threads < 1:
        raise ValueError("invalid shift dimensions or thread count")
    for i in prange(y.shape[0], nogil=True, num_threads=n_threads,
                    schedule='static', use_threads_if=(y.shape[0] * y.shape[1] >= 200000)):
        for j in range(y.shape[1]):
            y[i, j] = y[i, j] - alpha * q[i, j]


def mask_seen(
    const real_t[::1] data, const index_t[::1] indices,
    const index_t[::1] indptr, real_t[:, ::1] scores,
    Py_ssize_t start, int n_threads=1
):
    cdef Py_ssize_t i, p, col, end = start + scores.shape[1]
    if scores.shape[0] + 1 != indptr.shape[0] or n_threads < 1:
        raise ValueError("invalid mask dimensions or thread count")
    for i in prange(scores.shape[0], nogil=True, num_threads=n_threads,
                    schedule='static', use_threads_if=(scores.shape[0] >= 128)):
        for p in range(indptr[i], indptr[i + 1]):
            col = indices[p]
            if start <= col < end and data[p] != 0:
                scores[i, col - start] = -INFINITY

# Algorithm and implementation

## Objective and orientation

With X of shape m × n and k components, the public factorization is
`Z = X @ C.T`, `X_hat = Z @ C`, with approximately orthonormal rows C.
The rank-k reconstruction objective includes implicit zeros and is not a
missing-entry-only objective. Input is not centred.

The core is an independent implementation of fixed-iteration **dashSVD Algorithm 4**
([Feng et al., 2024](https://arxiv.org/html/2404.09276v1)), with guarded normalization
and sklearn projection semantics. For m >= n, maintain an n × ell right basis:

```python
ell = min(k + oversampling, m, n)
Q = orth(X.T @ gaussian(m, ell))
alpha = 0
for iteration in range(n_iter):
    Y = X.T @ (X @ Q) - alpha * Q
    Q, shifted_singular_values, _ = thin_svd(Y)
    s_min = shifted_singular_values[-1]
    if s_min > alpha:
        alpha = (s_min + alpha) / 2
B = X @ Q
_, s, R_transpose = thin_svd(B)
C = (Q @ R_transpose[:k].T).T
Z = B @ R_transpose[:k].T
```

This pseudocode omits input validation, finite checks, sign fixing, scaling and
optional early stopping; the actual implementation includes them. No full X.T @ X
is materialized. The shift is **not** a fixed user-tuned damping factor. Its update
comes from the smallest retained singular value of the current shifted sketch.
The first applied shift is zero, so q=0 and q=1 cannot benefit from shifting.

For m < n, auto swaps X/X.T and maintains an m × ell left basis. It explicitly
computes the final public projection X @ C.T. Returning an approximate U*S from
the range approximation here would violate transform consistency. On the right
path, reusing B gives that projection without another sparse matrix product.

Right orientation costs `2*q+2` full-width matrix products during fit_transform;
left orientation costs `2*q+3`. This counter is not all memory traversals: validation,
transpose conversion and variance statistics also scan data. Main working storage
is O(nnz + (m+n)*ell + ell²), with constant-factor copies for the transpose and guarded normalization, plus O(n) float64 variance work arrays. Full reconstruction is not stored. Dense
input itself naturally costs O(m*n).

## Guarded small decompositions

Tall skinny sketches use a float64 Gram matrix even when working factors are
float32. Symmetric eigendecomposition yields the small rotation and singular values.
The fast path is rejected when the minimum/maximum Gram eigenvalue ratio is at most
1e-8 for float32 work or 1e-10 for float64 work, or the values are zero/nonfinite.
The fallback uses Householder QR followed by SVD of its small R; it does not invert
zero or tiny singular values. `normalizer="qr"` always uses that path.

The guards avoid many failures of naive Gram-SVD but do not undo precision already
lost while forming X.T @ (X @ Q). This is a speed-oriented approximate solver,
not a substitute for a high-accuracy residual-controlled SVD. Unusually large or
small input magnitudes are rescaled and outputs restored to original units.

## Optional stopping

`tol=0` disables stopping. With positive tol, successive estimates derived from
shifted sketch singular values are compared, normalizing by direction k+1.
The estimate uses the shift **actually applied** before computing the next shift.
This is a heuristic stopping signal, not a certified PVE bound or guaranteed
Frobenius/spectral residual. No timing table in this release uses early stopping.

## Native implementation

`_kernels.pyx` uses const typed memoryviews, float32/float64 and int32/int64 fused
specializations, `nogil` and `prange`. Input indices are checked once in the public
Python entry point. A row belongs to one thread, avoiding atomics, shared row
accumulators and nondeterministic cross-thread summation. The guided schedule helps
with uneven row lengths. Small products skip threading overhead.

`_simd.h` provides portable baseline row kernels and, on supported GCC/Clang x86-64
builds, runtime-selected AVX2/FMA and AVX-512/FMA functions. The selected function
pointer is called once per row, not per nonzero. Every nonzero updates a contiguous
sketch row; the inner factor dimension can therefore vectorize. Selection checks
CPU support and can be overridden before import using `MYMF_SIMD`.

GCC's outer-loop unroll-and-jam transformation scalarized the SIMD loop in an
examined build. The GCC-only `no-loop-unroll-and-jam` function attribute disables
that transformation while leaving the rest of `-O3` in place. This is a compiler-
specific implementation choice, not an algorithmic complexity improvement.
No unsafe `-ffast-math` flag is enabled. FMA/SIMD can still change rounding.

Dense matrix products, small rotations, QR, SVD and symmetric eigensolvers use
NumPy/SciPy BLAS/LAPACK rather than replacing vendor libraries with scalar C loops.


## v0.2 execution changes (not a new factorization algorithm)

For short dense rows, `_reg.h` keeps one to eight vector accumulators in registers
through the sparse row and stores only the final sums. Float32 covers widths up
to 128 with AVX-512 (64 with AVX2); float64 covers 64/32. Larger widths retain the
previous generic kernel. Final vector loads/stores are masked; no padding changes
the sketch width or overruns row buffers. The cases are generated by
`tools/generate_simd.py` and are independent of dataset dimensions and seeds.

Single-thread products bypass OpenMP entirely. Input CSR validity is still checked
at every public entry, but trusted scalar scaling and transpose results are not
revalidated redundantly. Positive explicit thread counts skip CPU-quota discovery.
`ThreadpoolController` caches library discovery; limits still enter/exit each call.
The lazy first discovery is included in the separate cold-fit measurements.

Integer seeds use PCG64 and float32 sketches use native float32 Gaussian generation.
The Gaussian law and requested sketch dimensions are unchanged; old exact seeds are
available through an explicit RandomState. Small Gram matrices call LAPACK DSYEVD
without repeated high-level batch/driver dispatch and retain the conditioning guards.

Column Welford moments over stored values are merged with the implicit-zero group
to compute total population variance. The count, mean and second-moment work arrays
are O(n_features); a float64 copy of all nnz values is no longer required.

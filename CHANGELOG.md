# Changelog

## 0.2.0

- Reduce per-fit overhead and add register-blocked SIMD row kernels with safe tails.
- Preserve the shifted-SVD objective, requested rank/width, full variance attributes,
  input validation and QR/SVD fallbacks; do not delegate factorization to sklearn.
- Reuse threadpool discovery, bypass serial OpenMP and redundant CSR scans.
- Generate integer-seeded sketches with PCG64 in the working dtype. This changes
  integer-seeded factors; pass RandomState for the legacy stream. Accept Generator.
- Add stable Cython sparse variance statistics and direct small-matrix LAPACK dispatch.
- Add real MovieLens 100K full-catalog evaluation with 300 fresh fits per method,
  AVX2-only checks, cold-process tests, and an optional assert-faster gate.
- Add masked-width, RNG, controller, variance, malformed-input and no-result-cache tests.
- Update the article while retaining the Alice TF-IDF similar-sentence example.

## 0.1.0 — 2026-09-27

Initial implementation: dynamic-shift randomized truncated SVD, guarded Gram
normalization, Cython/OpenMP sparse kernels with runtime SIMD dispatch, sklearn-style
`MyMF`, blocked recommendation, tests and reproducible offline benchmarks.

This is a source release, not a claim of publication to PyPI or a universal SOTA win.

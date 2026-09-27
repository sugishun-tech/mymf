"""Small-matrix decompositions with a guarded Gram fast path."""
from __future__ import annotations

import numpy as np
from scipy import linalg


def thin_svd(a: np.ndarray, *, normalizer: str = "auto", left: bool = True):
    """Return (U or None, s, Vt, used_fallback) for a tall-and-skinny matrix.

    Gram is accumulated in float64 even for float32 working factors. Ill-conditioned
    and rank-deficient cases use Householder QR followed by an SVD of the small R.
    No inverse singular values are formed in that fallback.
    """
    if not np.isfinite(a).all():
        raise FloatingPointError("Non-finite intermediate. Use dtype='float64'.")
    fallback = normalizer == "qr"
    if not fallback:
        work = np.asarray(a, dtype=np.float64)
        gram = work.T @ work
        del work
        eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False, driver="evd")
        # Squaring the condition number is unsafe for tiny singular directions.
        threshold = 1e-8 if a.dtype == np.float32 else 1e-10
        fallback = (
            not np.isfinite(eigenvalues).all()
            or eigenvalues[-1] <= 0
            or eigenvalues[0] <= threshold * eigenvalues[-1]
        )
        if not fallback:
            s64 = np.sqrt(eigenvalues[::-1])
            rotation = np.ascontiguousarray(eigenvectors[:, ::-1], dtype=a.dtype)
            u = None
            if left:
                weights = np.ascontiguousarray(eigenvectors[:, ::-1] / s64, dtype=a.dtype)
                u = np.ascontiguousarray(a @ weights)
            return u, s64.astype(a.dtype), rotation.T.copy(), False
    q, r = linalg.qr(a, mode="economic", check_finite=False)
    ur, s, vt = linalg.svd(r, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    u = np.ascontiguousarray(q @ ur) if left else None
    return u, s, np.ascontiguousarray(vt), True

"""Reuse library discovery, never cache data-dependent computation."""
from __future__ import annotations

from threading import Lock
from threadpoolctl import ThreadpoolController

_controller = None
_lock = Lock()


def _get_controller():
    global _controller
    if _controller is None:
        with _lock:
            if _controller is None:
                # NumPy and SciPy are already imported by the estimator.
                _controller = ThreadpoolController()
    return _controller


def blas_context(limits: int):
    return _get_controller().limit(limits=limits, user_api="blas")


def refresh_threadpools():
    """Rediscover native libraries loaded after the first MyMF call.

    Call outside concurrent fits. As with threadpoolctl, thread limits are
    process-wide. For concurrent use, set external limits and blas_threads=None.
    No fitted factors, matrix structure or input values are cached.
    """
    global _controller
    with _lock:
        _controller = ThreadpoolController()

"""CPU-first randomized matrix factorization."""
from .estimator import MyMF
from ._kernels import build_info

__version__ = "0.1.0"
__all__ = ["MyMF", "build_info", "__version__"]

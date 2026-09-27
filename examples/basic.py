"""Run after installing mymf: python examples/basic.py."""
import numpy as np
from scipy import sparse
from mymf import MyMF, build_info

X = sparse.random(2000, 800, density=0.015, format="csr",
                  dtype=np.float32, random_state=42)
model = MyMF(32, n_iter=2, random_state=42, n_jobs=2)
Z = model.fit_transform(X)
np.testing.assert_allclose(Z, model.transform(X), atol=3e-5, rtol=3e-5)
print("build:", build_info())
print("embedding:", Z.shape, Z.dtype)
print("components:", model.components_.shape)
print("matrix products:", model.matrix_passes_)
print("variance ratio sum:", model.explained_variance_ratio_.sum())

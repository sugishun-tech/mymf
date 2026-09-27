"""MyMF can be cloned and composed with sklearn estimators."""
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from mymf import MyMF

X = sparse.random(500, 200, density=.04, format="csr", random_state=42)
pipeline = make_pipeline(MyMF(16, random_state=42), Normalizer())
Z = pipeline.fit_transform(X)
print(Z.shape, pipeline.get_params()["mymf__n_components"])

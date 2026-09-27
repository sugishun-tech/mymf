"""Toy binary user-item data; this demonstrates the API, not recommendation quality."""
import numpy as np
from scipy.sparse import csr_matrix
from mymf import MyMF

X = csr_matrix(np.array([
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1],
], dtype=np.float32))
model = MyMF(2, random_state=42).fit(X)
ids, scores = model.recommend(X, n_recommendations=2,
                              batch_size=2, item_batch_size=3)
for user, (items, values) in enumerate(zip(ids, scores)):
    print(f"user={user}, recommended item columns={items}, scores={values}")
    assert all(item < 0 or X[user, item] == 0 for item in items)

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mymf import MyMF


# 1. Alice in Wonderland を読み込む
with open("alice_in_wonder_land.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 改行を空白にしてから、. ! ? を文末として大まかに文分割する。
# 厳密な英文分割器ではないが、サンプル用途ならこれで十分。
text = re.sub(r"\s+", " ", text).strip()
sentences = [
    s.strip()
    for s in re.split(r"(?<=[.!?])\s+", text)
    if len(s.strip()) >= 20
]

# 2. 文 × 単語 のTF-IDF疎行列を作る
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    min_df=2,
    dtype=np.float32,
)
X = vectorizer.fit_transform(sentences)

print("TF-IDF:", X.shape, "nnz=", X.nnz)

# 3. MyMFで低ランク化する
# n_componentsは入力サイズを超えないように調整する。
n_components = min(128, X.shape[0] - 1, X.shape[1] - 1)
if n_components < 1:
    raise ValueError("文または語彙が少なすぎて行列分解できません")

mf = MyMF(
    n_components=n_components,
    n_iter=2,
    n_oversamples=10,
    random_state=42,
    n_jobs=-1,
    blas_threads=1,
)

# 各文を n_components 次元の潜在ベクトルへ変換
Z = mf.fit_transform(X)
print("latent:", Z.shape)

# 4. index=200 の文に近い文 Top-k を探す
index = 200
k = 10

if index >= len(sentences):
    raise IndexError(
        f"index={index} だが、分割後の文数は {len(sentences)} しかありません"
    )

# 潜在空間上のcosine similarityを計算
similarities = cosine_similarity(
    Z[index:index + 1],
    Z,
).ravel()

# 自分自身は除外
similarities[index] = -np.inf

# 全件sortせず、まずTop-k候補だけ拾う
k = min(k, len(sentences) - 1)
topk = np.argpartition(similarities, -k)[-k:]
topk = topk[np.argsort(similarities[topk])[::-1]]

print(f"\nquery [{index}]")
print(sentences[index])

print(f"\nTop {k} similar sentences")
for rank, i in enumerate(topk, 1):
    print(f"{rank:2d}. score={similarities[i]:.4f} index={i}")
    print(f"    {sentences[i]}")

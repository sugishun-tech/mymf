#!/usr/bin/env python3
"""Full-catalog Recall/NDCG@20; synthetic by default, optional real MovieLens 100K."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from mymf import MyMF


def load_data(movielens):
    if movielens:
        # Official u.data columns: user, item, rating, timestamp. IDs start at 1.
        path = Path(movielens)
        records = np.loadtxt(path, dtype=np.int64)
        positive = records[records[:, 2] >= 4]
        users, items, timestamp = positive[:, 0] - 1, positive[:, 1] - 1, positive[:, 3]
        shape = (int(records[:, 0].max()), int(records[:, 1].max()))
        metadata = dict(name='MovieLens 100K positive ratings >= 4', synthetic=False,
                        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        # Independent taste groups, random interactions, then a held-out sample.
        rng = np.random.default_rng(2026)
        m, n, groups = 2400, 1600, 20
        users_list, items_list = [], []
        for user in range(m):
            group = int(rng.integers(groups))
            preferred = rng.choice(np.arange(group * 80, (group + 1) * 80),
                                   size=40, replace=False)
            outside = np.setdiff1d(np.arange(n), np.arange(group * 80, (group + 1) * 80))
            other = rng.choice(outside, size=10, replace=False)
            selected = rng.permutation(np.concatenate((preferred, other)))
            users_list.extend([user] * len(selected))
            items_list.extend(selected.tolist())
        users, items = np.asarray(users_list), np.asarray(items_list)
        timestamp = np.tile(np.arange(50), m)
        shape = (m, n)
        metadata = dict(name='Synthetic grouped preferences, not real recommendation evidence',
                        synthetic=True)
    train_rows, train_cols = [], []
    heldout = {}
    # Latest 20% of positives per user. Synthetic timestamps only randomize the split.
    for user in range(shape[0]):
        index = np.flatnonzero(users == user)
        if len(index) == 0:
            continue
        index = index[np.lexsort((items[index], timestamp[index]))]
        n_test = max(1, int(np.ceil(len(index) * .2))) if len(index) >= 5 else 0
        split = len(index) - n_test
        train_rows.extend([user] * split)
        train_cols.extend(items[index[:split]].tolist())
        if n_test:
            heldout[user] = set(items[index[split:]].tolist())
    train = sparse.csr_matrix((np.ones(len(train_rows), dtype=np.float32),
                               (train_rows, train_cols)), shape=shape)
    train.sum_duplicates()
    train.data[:] = 1
    metadata.update(shape=list(shape), train_nnz=int(train.nnz),
                    test_positives=sum(map(len, heldout.values())), evaluated_users=len(heldout),
                    protocol='per-user latest 20% of positives; users with >=5 positives; '
                             'all items except train positives are candidates; no sampled negatives')
    return train, heldout, metadata


def evaluate(model, train, heldout, cutoff=20):
    recall, ndcg = [], []
    user_ids = np.array(sorted(heldout))
    for start in range(0, len(user_ids), 128):
        batch_ids = user_ids[start:start + 128]
        query = train[batch_ids]
        scores = model.transform(query) @ model.components_
        for i, user in enumerate(batch_ids):
            seen = query.indices[query.indptr[i]:query.indptr[i + 1]]
            scores[i, seen] = -np.inf
            # Stable item-ID tiebreak, identical scoring for every model.
            top = np.argsort(-scores[i], kind='stable')[:cutoff]
            hits = np.array([item in heldout[int(user)] for item in top], dtype=np.float64)
            recall.append(float(hits.sum() / len(heldout[int(user)])))
            discounts = 1 / np.log2(np.arange(2, cutoff + 2))
            ideal = discounts[:min(cutoff, len(heldout[int(user)]))].sum()
            ndcg.append(float(np.dot(hits, discounts) / ideal))
    return dict(recall_at_20=float(np.mean(recall)), ndcg_at_20=float(np.mean(ndcg)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--movielens', help='Local path to the official MovieLens 100K u.data file')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--output', default='benchmarks/results/recommendation.json')
    args = p.parse_args()
    if args.threads < 1 or args.repeat < 1:
        p.error('threads and repeat must be positive')
    train, heldout, metadata = load_data(args.movielens)
    if not heldout:
        p.error('no users meet the evaluation criteria')
    names = ['sklearn_q2', 'sklearn_q5', 'mymf_q0', 'mymf_q2', 'mymf_q4', 'mymf_q5']
    results = {name: [] for name in names}
    with threadpool_limits(1):
        # A warmup is not part of any measured fit.
        for cls in (MyMF, TruncatedSVD):
            cls(32, random_state=0).fit(train)
        for seed in range(args.repeat):
            for name in np.random.default_rng(seed).permutation(names):
                q = int(name[-1])
                common = dict(n_components=32, n_iter=q, n_oversamples=10, random_state=seed)
                model = (MyMF(n_jobs=args.threads, blas_threads=1, **common)
                         if name.startswith('mymf') else TruncatedSVD(**common))
                t = time.perf_counter()
                model.fit_transform(train)
                fit_time = time.perf_counter() - t
                metric = evaluate(model, train, heldout)
                results[name].append(dict(seed=seed, fit_seconds=fit_time, **metric))
                print(name, seed, fit_time, metric, flush=True)
    summary = {name: {key: float(np.mean([r[key] for r in records]))
                      for key in ['fit_seconds', 'recall_at_20', 'ndcg_at_20']}
               for name, records in results.items()}
    for name, records in results.items():
        summary[name]['fit_seconds_median'] = float(np.median([r['fit_seconds'] for r in records]))
        summary[name]['recall_at_20_std'] = float(np.std([r['recall_at_20'] for r in records]))
        summary[name]['ndcg_at_20_std'] = float(np.std([r['ndcg_at_20'] for r in records]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(dataset=metadata, parameters=vars(args),
                                     records=results, summary=summary), indent=2) + '\n')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()

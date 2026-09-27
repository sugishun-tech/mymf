#!/usr/bin/env python3
"""Full-catalog Recall/NDCG@20; synthetic by default, optional real MovieLens 100K."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import platform
import sys

import scipy
import sklearn
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits, threadpool_info

from mymf import MyMF, build_info, __version__


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



def environment():
    cpuinfo = Path('/proc/cpuinfo')
    cpu = cpuinfo.read_text() if cpuinfo.exists() else ''
    model = next((line.split(':', 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith('model name')), platform.processor())
    return dict(python=sys.version, platform=platform.platform(), cpu=model,
                mymf=__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, build=build_info(),
                threadpools=threadpool_info())


def make_model(name, rank, oversamples, threads, seed):
    q = int(name.rsplit('q', 1)[1])
    common = dict(n_components=rank, n_iter=q, n_oversamples=oversamples,
                  random_state=seed)
    if name.startswith('mymf'):
        if 'legacy' in name:
            common['random_state'] = np.random.RandomState(seed)
        return MyMF(n_jobs=threads, blas_threads=1, **common)
    return TruncatedSVD(**common)


def reconstruction_metrics(model, train):
    # Use the actual X @ components.T projection, not the surrogate U*S.
    c = model.components_.astype(np.float64)
    z = train @ c.T
    norm2 = np.dot(train.data.astype(np.float64), train.data.astype(np.float64))
    ztz, cct = z.T @ z, c @ c.T
    residual = max(0., float(norm2 - 2 * np.trace(ztz) + np.sum(ztz * cct)))
    return dict(relative_frobenius_error=float(np.sqrt(residual / norm2)),
                orthogonality_error=float(np.linalg.norm(cct - np.eye(c.shape[0]))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--movielens', help='Local path to the official MovieLens 100K u.data file')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--repeat', type=int, default=10, help='Number of independent sketch seeds')
    p.add_argument('--timing-repeats', type=int, default=30,
                   help='Fresh fits per seed and method; includes every fit, not just the fastest')
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--oversamples', type=int, default=10)
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--include-legacy', action='store_true',
                   help='Also measure v0.2 with an explicit legacy RandomState (no RNG change)')
    p.add_argument('--assert-faster', action='store_true',
                   help='After saving results, fail if MyMF does not beat sklearn for q=2 AND q=5')
    p.add_argument('--output', default='benchmarks/results/recommendation.json')
    args = p.parse_args()
    if min(args.threads, args.repeat, args.timing_repeats, args.rank) < 1 or args.oversamples < 0:
        p.error('threads, repeats and rank must be positive; oversamples must be nonnegative')
    train, heldout, metadata = load_data(args.movielens)
    train = train.astype(args.dtype, copy=False)
    metadata['dtype'] = str(train.dtype)
    if not heldout:
        p.error('no users meet the evaluation criteria')
    names = ['sklearn_q2', 'sklearn_q5', 'mymf_q0', 'mymf_q2', 'mymf_q4', 'mymf_q5']
    if args.include_legacy:
        names += ['mymf_legacy_q2', 'mymf_legacy_q5']
    results = {name: [] for name in names}
    with threadpool_limits(1):
        env = environment()
        # Warm every exact method once. No input-dependent results are reused.
        for name in names:
            make_model(name, args.rank, args.oversamples, args.threads, 19).fit_transform(train)
        ordering = np.random.default_rng(90210)
        for seed in range(args.repeat):
            times = {name: [] for name in names}
            final_models = {}
            for trial in range(args.timing_repeats):
                for name in ordering.permutation(names):
                    model = make_model(name, args.rank, args.oversamples, args.threads, seed)
                    t = time.perf_counter_ns()
                    z = model.fit_transform(train)
                    fit_time = (time.perf_counter_ns() - t) * 1e-9
                    times[name].append(fit_time)
                    final_models[name] = model
                    del z
            # Never mix quality-scoring time into the fit measurements.
            for name in names:
                model = final_models[name]
                metric = evaluate(model, train, heldout)
                metric.update(reconstruction_metrics(model, train))
                elapsed = float(np.median(times[name]))
                results[name].append(dict(seed=seed, fit_seconds=elapsed,
                                         fit_seconds_samples=times[name], **metric))
                print(name, seed, f'{elapsed * 1000:.3f} ms', metric, flush=True)
    summary = {}
    for name, records in results.items():
        times = np.array([t for r in records for t in r['fit_seconds_samples']])
        summary[name] = dict(fit_seconds=float(times.mean()),
                             fit_seconds_median=float(np.median(times)),
                             fit_seconds_p10=float(np.quantile(times, .1)),
                             fit_seconds_p90=float(np.quantile(times, .9)),
                             timed_fits=int(times.size))
        for key in ['recall_at_20', 'ndcg_at_20', 'relative_frobenius_error',
                    'orthogonality_error']:
            values = [r[key] for r in records]
            summary[name][key] = float(np.mean(values))
            summary[name][key + '_std'] = float(np.std(values))
    speedups = {f'q{q}': summary[f'sklearn_q{q}']['fit_seconds_median']
                         / summary[f'mymf_q{q}']['fit_seconds_median'] for q in (2, 5)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(dataset=metadata, parameters=vars(args), environment=env,
                  timing_scope='fit_transform including validation, random sketch generation, '
                               'transpose preparation, decomposition, projection, variance and '
                               'internal BLAS thread limiting; excluding imports, dataset loading, '
                               'model construction, warmup and quality evaluation; fresh model per fit',
                  records=results, summary=summary, sklearn_over_mymf_speedup=speedups,
                  rng_note='Integer seeds in MyMF v0.2 use PCG64; sklearn uses RandomState. '
                           'The optional legacy variants use explicit RandomState for MyMF. '
                           'Same seed numbers do not imply identical sketches across implementations.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {output}; speedups={speedups}', flush=True)
    if args.assert_faster and any(ratio <= 1.0 for ratio in speedups.values()):
        raise SystemExit('Speed check failed on this machine. Results have been saved, not discarded.')


if __name__ == '__main__':
    main()

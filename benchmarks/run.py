#!/usr/bin/env python3
"""Reproducible, offline timing/quality benchmark. Run from any working directory."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_info, threadpool_limits

from mymf import MyMF, build_info


def clustered_sparse(m, n, per_row=40, groups=32, seed=42):
    """Positive block structure plus 25% uniform noise; values are event counts."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(m), per_row)
    labels = rng.integers(groups, size=m)
    items_per_group = n // groups
    cols = labels[rows] * items_per_group + rng.integers(items_per_group, size=rows.size)
    noise = rng.random(rows.size) < .25
    cols[noise] = rng.integers(n, size=int(noise.sum()))
    x = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(m, n))
    x.sum_duplicates()
    x.sort_indices()
    return x


def dense_spectrum(m=1200, n=600, seed=42):
    """Known spectrum; no reference solver or reference approximation is needed."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(m, n)))
    v, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.geomspace(50.0, .05, n)
    return np.ascontiguousarray((u * s) @ v.T, dtype=np.float64), s


def models(k, threads, blas_threads, seed):
    common = dict(n_components=k, n_oversamples=10, random_state=seed)
    return {
        'sklearn_q2': TruncatedSVD(n_iter=2, **common),
        'sklearn_q5': TruncatedSVD(n_iter=5, **common),
        'mymf_q0': MyMF(n_iter=0, n_jobs=threads, blas_threads=blas_threads, **common),
        'mymf_q2': MyMF(n_iter=2, n_jobs=threads, blas_threads=blas_threads, **common),
        'mymf_q4': MyMF(n_iter=4, n_jobs=threads, blas_threads=blas_threads, **common),
        'mymf_q5': MyMF(n_iter=5, n_jobs=threads, blas_threads=blas_threads, **common),
        'mymf_unshifted_q2': MyMF(n_iter=2, algorithm='randomized', n_jobs=threads,
                                 blas_threads=blas_threads, **common),
    }


def environment(args):
    def read(path):
        return Path(path).read_text().strip() if Path(path).exists() else None
    cpu = read('/proc/cpuinfo') or ''
    model = next((line.split(':', 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith('model name')), platform.processor())
    return dict(python=sys.version, platform=platform.platform(), cpu=model,
                cpu_max=read('/sys/fs/cgroup/cpu.max'), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__,
                build=build_info(), threadpools=threadpool_info(),
                parameters=vars(args))


def quality(x, estimator, exact_s=None):
    """Compute the actual public projection, rather than trusting U*S."""
    c = estimator.components_.astype(np.float64)
    z = x @ c.T
    if sparse.issparse(x):
        norm2 = np.dot(x.data.astype(np.float64), x.data.astype(np.float64))
    else:
        norm2 = np.einsum('ij,ij->', x, x, dtype=np.float64)
    # Valid even when components have a small orthogonality error:
    # ||X-ZC||_F^2 = ||X||_F^2 - 2||Z||_F^2 + tr((Z^T Z)(C C^T)).
    ztz = z.T @ z
    cct = c @ c.T
    captured = float(np.trace(ztz))
    residual2 = max(0., float(norm2 - 2 * captured + np.sum(ztz * cct)))
    metrics = dict(projection_energy=captured / norm2,
                   relative_frobenius_error=float(np.sqrt(residual2 / norm2)),
                   orthogonality_error=float(np.linalg.norm(cct - np.eye(c.shape[0]))))
    if exact_s is not None:
        optimum2 = float(np.sum(exact_s[c.shape[0]:] ** 2))
        metrics['error_over_optimal'] = float(np.sqrt(residual2 / optimum2))
        metrics['top_k_singular_value_relative_error'] = float(np.linalg.norm(
            estimator.singular_values_ - exact_s[:c.shape[0]])
            / np.linalg.norm(exact_s[:c.shape[0]]))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['sparse_medium', 'sparse_large', 'dense'],
                   default='sparse_medium')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--blas-threads', type=int, default=1)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--output', default='benchmarks/results/local.json')
    args = p.parse_args()
    if min(args.threads, args.blas_threads, args.repeat, args.rank) < 1:
        p.error('thread counts, repeats and rank must be positive')
    with threadpool_limits(args.blas_threads):
        if args.case == 'dense':
            x, exact_s = dense_spectrum()
        else:
            m, n, per_row = (20000, 8000, 40) if args.case == 'sparse_medium' else (50000, 20000, 50)
            x = clustered_sparse(m, n, per_row)
            exact_s = None
        names = list(models(args.rank, args.threads, args.blas_threads, 0))
        records = {name: dict(times=[], seeds=[], quality=[]) for name in names}
        # Warm up every method once. Compilation and data generation are not timed.
        for model in models(args.rank, args.threads, args.blas_threads, 0).values():
            model.fit_transform(x)
        rng = np.random.default_rng(2026)
        for repetition in range(args.repeat):
            seed = 100 + repetition
            for name in rng.permutation(names):
                gc.collect()
                model = models(args.rank, args.threads, args.blas_threads, seed)[name]
                started = time.perf_counter()
                z = model.fit_transform(x)
                elapsed = time.perf_counter() - started
                del z
                metric = quality(x, model, exact_s)
                record = records[name]
                record['times'].append(elapsed)
                record['seeds'].append(seed)
                record['quality'].append(metric)
                record['fallbacks'] = getattr(model, 'n_fallbacks_', None)
                record['matrix_passes'] = getattr(model, 'matrix_passes_', None)
                print(f'{args.case:14} {name:21} repeat={repetition} '
                      f'{elapsed:.4f}s error={metric["relative_frobenius_error"]:.6f}', flush=True)
        for record in records.values():
            record['seconds_median'] = float(np.median(record['times']))
            record['seconds_min'] = float(np.min(record['times']))
            record['seconds_max'] = float(np.max(record['times']))
            record['quality_mean'] = {key: float(np.mean([q[key] for q in record['quality']]))
                                      for key in record['quality'][0]}
            record['quality_std'] = {key: float(np.std([q[key] for q in record['quality']]))
                                     for key in record['quality'][0]}
        result = dict(environment=environment(args), dataset=dict(
            name=args.case, synthetic=True, shape=list(x.shape), dtype=str(x.dtype),
            nnz=int(x.nnz) if sparse.issparse(x) else int(np.count_nonzero(x)),
            input_bytes=int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)
            if sparse.issparse(x) else int(x.nbytes)), records=records,
            timing_scope='fit_transform including conversion, validation, transpose and variance; '
                         'excluding generation, warmup and quality scoring')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {destination}')


if __name__ == '__main__':
    main()

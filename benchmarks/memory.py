#!/usr/bin/env python3
"""Fresh-process total peak RSS, including input/runtime. Linux measurements only."""
import argparse
import json
import resource
import time

from threadpoolctl import threadpool_limits
from sklearn.decomposition import TruncatedSVD
from mymf import MyMF
from run import clustered_sparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', choices=['mymf', 'sklearn'], required=True)
p.add_argument('--iterations', type=int, default=2)
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
x = clustered_sparse(50000, 20000, 50)
common = dict(n_components=64, n_iter=a.iterations, random_state=42)
model = MyMF(n_jobs=a.threads, **common) if a.model == 'mymf' else TruncatedSVD(**common)
with threadpool_limits(1):
    start = time.perf_counter()
    z = model.fit_transform(x)
    elapsed = time.perf_counter() - start
print(json.dumps(dict(model=a.model, iterations=a.iterations, seconds=elapsed,
                     process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     note='Linux ru_maxrss, total process peak, not incremental algorithm memory')))

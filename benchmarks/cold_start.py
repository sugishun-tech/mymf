#!/usr/bin/env python3
"""First fit in fresh Python processes; imports and dataset I/O are not timed."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from recommendation import load_data, make_model, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--movielens', required=True)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--output', default='benchmarks/results/cold_start.json')
    p.add_argument('--child', choices=['sklearn_q2', 'sklearn_q5', 'mymf_q2', 'mymf_q5'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.child:
        x, _, metadata = load_data(args.movielens)
        model = make_model(args.child, 32, 10, 1, args.seed)
        with threadpool_limits(1):
            t = time.perf_counter_ns()
            model.fit_transform(x)
            seconds = (time.perf_counter_ns() - t) * 1e-9
            env = environment()
        print(json.dumps(dict(seed=args.seed, fit_seconds=seconds, dataset=metadata,
                              environment=env)))
        return
    if args.repeat < 1:
        p.error('repeat must be positive')
    names = ['sklearn_q2', 'sklearn_q5', 'mymf_q2', 'mymf_q5']
    results = {name: [] for name in names}
    order = np.random.default_rng(230)
    env, metadata = None, None
    for trial in range(args.repeat):
        for name in order.permutation(names):
            command = [sys.executable, str(Path(__file__).resolve()), '--child', name,
                       '--movielens', str(Path(args.movielens).resolve()), '--seed', str(trial)]
            r = json.loads(subprocess.check_output(command, text=True))
            env, metadata = r.pop('environment'), r.pop('dataset')
            results[name].append(r)
    summary = {name: dict(fit_seconds_median=float(np.median([r['fit_seconds'] for r in rs])))
               for name, rs in results.items()}
    payload = dict(dataset=metadata, environment=env, parameters=vars(args), records=results,
                   summary=summary, timing_scope='First fit_transform in a fresh interpreter; '
                   'no MyMF controller warmup; import and dataset loading time excluded for both; '
                   'both methods use one BLAS thread and MyMF n_jobs=1')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

# Reproducible measurements

See [../docs/benchmarks.md](../docs/benchmarks.md). The `results/v0.2/` directory
contains actual MovieLens 100K and new large-sparse measurements, including raw
fresh-fit timings, all seeds, environment, dataset SHA-256 and quality metrics.
`results/` at the top level contains historical v0.1 synthetic and memory results.
`results/v0.1/user_local_movielens.json` is the user's original v0.1 report from
another machine; do not compare absolute cross-machine timings as a speedup.

No dataset is bundled. Place the official u.data locally and run recommendation.py.
The default is ten seeds and thirty timed fresh fits per seed. Use `--assert-faster`
to fail after saving the JSON if q=2 or q=5 is not faster on your own machine.
`cold_start.py` tests the first fit in independently launched Python processes.

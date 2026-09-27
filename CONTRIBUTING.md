# Contributing

Create a virtual environment, install `-e '.[dev]'` and run `python -m pytest -q`.
Keep Cython/C kernels private and validate public inputs before unchecked native
access. Add numerical and malformed-input regression tests with each change.

Performance changes should include end-to-end timings and quality at both the same
iteration count and a matched-quality operating point. Record CPU/compiler/BLAS,
thread limits, data generation, seed, repeats and memory. Do not hide preprocessing
outside timers or present synthetic recommendation metrics as real-data results.

Avoid machine-specific global compiler flags in the default build. Never bundle
third-party datasets without respecting their licenses. Do not load untrusted pickle
files during tests or reports.

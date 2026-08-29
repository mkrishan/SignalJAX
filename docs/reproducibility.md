# Reproducibility guide

Use an isolated Python 3.10+ environment, install the project with `python -m pip install -e
'.[dev]'`, and run:

```bash
ruff format --check src tests examples
ruff check src tests examples
pytest --cov=kl_projection_message_passing --cov-report=term-missing --cov-fail-under=90
python examples/reproduce_paper_identities.py
python -m build
twine check dist/*
```

The compact report compares independent calculations wherever practical:

- explicit DAG pullbacks against a monolithic JAX gradient;
- tree sum-product marginals against dense joint enumeration;
- SPN upward-downward marginals against assignment enumeration;
- shared-DAG contexts against the sum of their unfolded tree copies;
- finite operator readouts against exact JAX differentials;
- a unit residual step against an independently constructed Euclidean projection;
- observed signal traces against separately supplied analytic contraction bounds.

Tests enable 64-bit JAX arithmetic for stringent identity checks. The library itself does not alter
global JAX settings, so applications remain responsible for their desired precision policy.

The exact enumeration and unfolding utilities are intended for small verification models. They are
not scalability benchmarks.

`requirements/tested-cpu.txt` records the exact CPU dependency versions used for the primary 0.2.0
local verification. The full test suite is also run against the declared minimum, JAX 0.4.30, on
Python 3.12. Continuous integration repeats that minimum-JAX check and tests the normal dependency
resolution on Python 3.10 and 3.12.

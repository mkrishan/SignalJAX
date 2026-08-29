# Contributing

Bug reports, mathematical counterexamples, and reproducibility reports are welcome.

This repository is maintained as the companion implementation for a specific paper. Changes must
preserve the distinction between deterministic differential readouts and probabilistic marginal
readouts, and every mathematical change must include a focused numerical test.

Manish Krishan Lal is the sole copyright owner of the project. Code contributions are accepted only
after a separate written contributor agreement assigning the contribution's copyright to Manish
Krishan Lal. Opening an issue or participating in technical discussion does not transfer copyright.

Development checks:

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest --cov=kl_projection_message_passing --cov-report=term-missing --cov-fail-under=90
python -m build
twine check dist/*
```

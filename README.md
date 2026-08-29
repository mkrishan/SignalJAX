# Kullback–Leibler Projection Message Passing: Backpropagation and Exact Marginals

SignalJAX studies signal propagation across gradient-based, belief-based, projection-based, and 
backpropagation-free paradigms through a modular message-passing calculus.

The accompanying arXiv revision is *Backpropagation from KL Projections: Differential and Exact I-Projection Correspondences*,
[arXiv:2512.24335](https://arxiv.org/abs/2512.24335).

The common experimental object is a lifted local operator together with a declared readout. This
separates the mechanism that moves state from the meaning of the signal being measured.

| Regime | State transition | Signal readout | Package claim |
|---|---|---|---|
| Deterministic computation | local smooth maps | cotangent/log-message differential | established at a fixed forward point |
| Tree BP and complete/decomposable SPNs | sum-product/KL projection | exact marginal | established under documented structure |
| Finite projection learning | repeated constraint projections | target or projection residual | finite operator instance; convergence is method-specific |
| Gaussian softening | conjugate sum-product sweep | posterior mean/Kalman-BLR step | established for the scalar conjugate tree |
| Predictive coding, target propagation, VMP, EP, max-product, feedback alignment, forward-only learning | method-supplied local maps | method-specific | modular comparison interface, not an equivalence theorem |

## What is implemented

- KL divergence, diagonal consensus I-projection, and product reverse-KL projection.
- Normalized finite factor-graph messages, exact tree schedules, loopy diagnostics, and dense
  verification.
- Explicit deterministic DAGs with local JAX pullbacks, arbitrary cotangent seeds, and path-additive
  readouts.
- Complete/decomposable SPNs with variable marginals, gate conditionals, contextual gate masses,
  node-indexed DAG unfolding, and copy merging.
- Composable finite operators with stage traces, relaxation, fixed-point residuals, JVPs, and
  finite-versus-differential comparisons.
- Independent Euclidean cyclic projections, affine/box/consensus projectors, and the standard
  squared-distance residual corollary.
- Signal studies for arbitrary JAX pytrees at node, edge, layer, block, iteration, or graph
  granularity.
- Explicit signal channels for cotangents, finite targets, projection residuals, marginals, energy
  errors, layerwise targets, local scores, and state perturbations.
- A claim-aware paradigm registry whose interface-only entries prevent exploratory connections from
  being mislabeled as proved equivalences.
- Scalar Gaussian finite and differential trajectories with exact one-step discrepancy and
  contraction factors.

Signal studies are not limited to a forward and backward pass. Any finite operator trajectory or
user-defined readout sequence can be measured. The package distinguishes:

1. observed amplification or attenuation in a run;
2. monotone non-increase of measured norms;
3. certified exponential attenuation, which is reported only when independent local bounds have a
   uniform maximum strictly below one.

## Installation

```bash
python -m pip install -e '.[dev]'
```

The package requires Python 3.10 or newer and JAX 0.4.30 or newer. It does not change JAX's global
precision configuration.

## Studies and examples

```bash
python examples/reproduce_paper_identities.py
python examples/signal_attenuation.py
python examples/projection_differential_bridge.py
python examples/paradigm_calculus.py
python examples/deterministic_readout.py
python examples/spn_marginals.py
python examples/gaussian_softening.py
```

The documentation explains the [paradigm calculus](docs/paradigm_calculus.md), the
[attenuation protocol](docs/signal_studies.md), and the
[reviewer-driven studies](docs/reviewer_questions.md).

## Verification

```bash
ruff format --check src tests examples
ruff check src tests examples
pytest --cov=kl_projection_message_passing --cov-report=term-missing --cov-fail-under=90
python -m build
twine check dist/*
```

Continuous integration tests Python 3.10 and 3.12 and separately runs the full suite against the
declared minimum JAX 0.4.30.

## Citation

Please cite the accompanying paper:

```bibtex
@article{lal2025backpropagation,
  author = {Manish Krishan Lal},
  title = {Backpropagation from KL Projections: Differential and Exact I-Projection Correspondences},
  year = {2025},
  eprint = {2512.24335},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2512.24335}
}
```

## Scientific boundaries

The package does not claim that all registered paradigms are the same algorithm. In particular, it
does not claim generic trajectory equivalence among projection learning, gradient descent,
predictive coding, target propagation, VMP, EP, or natural-gradient methods. Loopy, nonconvex,
nonsmooth, and inconsistent problems require their own convergence and stability analysis.

No implementation from the separately supplied projection-training work is incorporated. Its
unit-step Euclidean residual behavior is represented only by the standard mathematical corollary
`x - (x - P_C(x)) = P_C(x)`.

The code is internally modular. Future Modula or Modular/MAX integrations can implement the same
operator/readout interfaces without changing the scientific claim registry.

## License

Apache License 2.0. Copyright 2025 Manish Krishan Lal. See `LICENSE` and `NOTICE`.

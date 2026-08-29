# Mathematical scope and software claims

This software is named **“SignalJAX”** It accompanies arXiv:2512.24335 and turns the proved identities into independently
checkable numerical objects while adding a claim-aware platform for the broader signal-study program.

## Shared operation

The finite-table core exposes two operations:

1. diagonal restriction and normalization, representing agreement among replicas;
2. marginal matching into a product table, representing restoration of factorized structure.

Their composition is implemented as `consensus_product_operator`. It is a local finite-table model of
the consensus/product operator, not a claim that one dense joint tensor is an efficient representation
of a large region graph.

## Deterministic semantics

`DeterministicGraph` performs an explicit local reverse traversal. Every operation constructs its own
JAX pullback; contributions from multiple child paths are accumulated at their shared parent. An
output log-potential supplies the seed `grad(log(phi))`.

`deterministic_factor_message` constructs the local positive message
`m_x(x) = m_y(psi(x))` with other parents fixed. Differentiating its logarithm checks the local
delta-factor chain-rule identity directly. These are differential readouts at a fixed forward point,
not posterior marginals and not iterates of a training algorithm.

## Probabilistic semantics

`SumProductNetwork` accepts only topologically ordered, complete, decomposable circuits whose root
covers all declared variables. Its upward-downward pass returns:

- `lambda[i,t] * dS/dlambda[i,t] / S`, for variable marginals;
- `w[s,c] * S(c) / S(s)`, for conditional gate responsibilities;
- `D(s) * w[s,c] * S(c) / S(root)`, for contextual gate masses.

The downward traversal adds contributions from all parents of a shared DAG node. `unfold_spn` and
`merge_unfolded_downward` make the copy-and-merge identity testable without identifying distinct
same-scope mixture components.

## Positive Gaussian example

The scalar linear-Gaussian module computes the exact conjugate posterior mean and variance after a
softened deterministic relation. It also reports the narrow-belief differential approximation and
their one-step discrepancy. This is an exact Kalman/conjugate calculation on a tree. The software
does not extrapolate it to a general equivalence between projection, gradient, VMP, EP, predictive
coding, or target-propagation trajectories.

## Cross-paradigm signal studies

The generic operator, paradigm, and signal modules are research instrumentation rather than new
equivalence theorems. They make it possible to run the same norm/gain analysis on cotangents, finite
targets, residuals, marginals, energy errors, and local scores while retaining their distinct semantic
labels. An observed decreasing trace is not labeled exponentially attenuating unless independent
local contraction bounds certify a uniform factor below one.

The Euclidean projection-residual identity is included as a generic corollary. It does not extend the
gradient-of-distance interpretation to arbitrary geometries or optimizer-transformed residuals.

## Assumptions and failure modes

- KL projections and logarithmic readouts assume positive mass. Boundary behavior is explicit and is
  not silently repaired with arbitrary clipping.
- Exact generic BP is asserted only for connected trees. The loopy routine reports a numerical fixed
  point without claiming convergence or exactness.
- Exact SPN marginals require completeness and decomposability; malformed circuits are rejected.
- SPN enumeration and DAG unfolding are verification utilities and can grow exponentially.
- The package uses smooth JAX primitives. Nonsmooth generalized derivatives and implicit
  differentiation through fixed points are outside this release.
- Predictive coding, target propagation, VMP, EP, max-product, feedback alignment, and forward-only
  learning are executable interface slots. Their presence does not assert exact equivalence to the
  paper's KL operator.

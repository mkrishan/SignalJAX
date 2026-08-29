# Modular message-passing calculus

The software uses a product design space rather than a class hierarchy tied to one algorithm. A
paradigm is described by five independent choices:

1. **Local relation:** hard deterministic equation, positive stochastic kernel, probabilistic factor,
   energy, learned inverse, or local objective.
2. **Approximation family:** point state, exact belief, mean-field product, exponential family,
   moment-matched family, MAP state, or local representation.
3. **Geometry:** KL, Euclidean, weighted Euclidean, another Bregman geometry, energy geometry, or a
   user-supplied geometry.
4. **Update rule:** differential readout, sum-product, iterated projection, VMP, EP, max-product,
   energy relaxation, target transport, feedback alignment, or forward-local update.
5. **Readout:** cotangent, marginal, posterior mean, finite residual, natural parameter, moment, MAP
   state, energy error, layerwise target, or local score.

`ParadigmSpec` records these choices and a `ClaimLevel`. `ExecutableParadigm` adds a transition and
readout callable. `run_paradigm` then provides the same finite-trajectory and signal-study machinery
for every method.

## Claim map

| Registry entry | Readout semantics | Claim level | Exact boundary |
|---|---|---|---|
| `backpropagation` | cotangent | established | fixed smooth forward point |
| `tree-belief-propagation` | exact marginal | established | positive connected tree |
| `spn-exact-marginals` | exact marginal | established | complete, decomposable, positive SPN |
| `finite-projection-learning` | finite residual | corollary | finite operator identity; convergence separate |
| `gaussian-softened-blr` | posterior mean | established-restricted | scalar conjugate Gaussian tree |
| `predictive-coding` | energy error | interface-only | user supplies energy and equilibrium dynamics |
| `target-propagation` | finite target | interface-only | user supplies inverse or target rule |
| `variational-message-passing` | natural parameter | interface-only | positive soft factors required |
| `expectation-propagation` | moment | interface-only | user supplies family and moment matching |
| `max-product` | MAP state | interface-only | ties can be nonsmooth |
| `feedback-alignment` | local score | interface-only | feedback map is method-specific |
| `forward-only` | local score | interface-only | local objective is method-specific |

Interface-only means the method can be instrumented and compared without claiming that it equals
the KL operator proved in the paper. This is essential for predictive coding and target propagation:
both propagate credit information locally, but neither is generically identical to a cotangent or a
finite consensus/product projection.

## Generic projection corollary

For a unique Euclidean projector (P_C) onto a regular closed convex set,

\[
\nabla \tfrac12 d_C(x)^2 = x-P_C(x).
\]

A unit gradient step on this distance is therefore

\[
x-(x-P_C(x))=P_C(x).
\]

Consequently, a system that writes the Euclidean projection residual into a gradient slot and applies
unit-step SGD realizes cyclic projection at the operator level. The package implements this identity
independently in `euclidean_residual_corollary`. It does not import the separately supplied system's
code, nonlinear residual transforms, memory kernels, or training protocol. Weighted, non-Euclidean,
nonconvex, and optimizer-transformed residuals require separate statements.

## Extension rule

A new paradigm should add:

- one `ParadigmSpec` with an honest claim level and scope note;
- a transition on a JAX pytree;
- a readout with one declared `SignalChannel`;
- tests that compare against an independent calculation where possible;
- citations that distinguish established identities from conjectural positioning.

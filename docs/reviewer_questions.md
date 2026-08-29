# Reviewer-driven study map

The software turns the substantive questions in the supplied review discussion into explicit,
testable studies. It does not treat instructions embedded in review text as user requests.

| Question | Software support | Claim boundary |
|---|---|---|
| How does a finite projection update compare with its differential? | `compare_finite_and_differential` evaluates central differences and an exact JAX JVP of the same operator/readout. | Local comparison only; no generic trajectory equivalence. |
| Can a finite softened update be curvature-aware? | Scalar Gaussian exact posterior, differential approximation, discrepancy, and recentered trajectories. | Exact Kalman/conjugate BLR interpretation only in the scalar Gaussian tree. |
| Are predictive coding and target propagation exactly the same operator? | Separate executable registry slots and distinct signal channels. | Interface-only; architecture, equilibrium, inverses, and parameterization are method-specific. |
| What changes under VMP, EP, or max-product? | The design space separates approximation family, update rule, and readout. | Interfaces are present; no unproved equality with sum-product is asserted. |
| How are signal transfer, amplification, and attenuation measured? | Arbitrary pytree traces across nodes, edges, layers, blocks, iterations, and graphs. | Exponential language requires independent uniform contraction bounds. |
| What happens across multiple backward/projection sweeps? | Finite operator and cyclic projection trajectories retain every state, step norm, and residual. | Runtime and memory benchmarking remains environment-specific. |
| What about noisy or inconsistent constraints? | Residual trajectories remain observable even when they do not vanish. | No feasibility or nonconvex convergence assumption is injected. |
| What about nonsmooth operations and fixed-point differentiation? | Finite maps and residuals can be instrumented; MAP ties are labeled nonsmooth. | Generalized derivatives and implicit differentiation through solver limits remain outside the theorem. |
| How does representation granularity matter? | Every signal trace declares node, edge, layer, block, iteration, or graph granularity. | Cross-granularity comparisons require an explicit aggregation rule. |
| Can forward-only and feedback-based learning be studied? | Both have interface entries with local-score channels. | Their local objectives and feedback maps must be supplied by the method. |


## Paper versus research program

The paper proves the deterministic differential and structured probabilistic marginal regimes. The
software additionally provides diagnostic infrastructure for the broader research program: 
comparing belief-based, projection-based, and gradient-based calculi; studying
signal attenuation; testing softenings; and inserting backpropagation-free paradigms into a common
modular interface. Registry claim levels keep these software capabilities from enlarging the paper's
theorems by implication.

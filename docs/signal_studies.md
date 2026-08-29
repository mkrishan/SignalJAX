# Signal amplification and attenuation studies

The signal-study layer is intentionally more general than a forward/backward analysis. It accepts
any ordered sequence of JAX pytrees and records what the sequence means.

## Channels

- **Cotangent:** reverse-mode differential at a fixed forward point.
- **Projection target:** finite displacement transported by a constraint method.
- **Projection residual:** (P_C(x)-x) or the residual of a composed operator.
- **Marginal:** exact or approximate probability-table readout.
- **Energy error:** local mismatch used by an energy or predictive-coding method.
- **Layerwise target:** finite target produced by target propagation or a related method.
- **Local score:** forward-only, feedback-alignment, or user-defined local learning signal.
- **State perturbation:** separation between two finite state trajectories.

Channels are never silently compared as though their units and semantics were identical.

## Axes and granularity

Signals can be ordered by node, edge, layer, block, operator iteration, or whole graph. Typical
studies include:

- JVP signal transfer from input to output;
- VJP/cotangent transfer from output to input;
- target transfer through projection layers;
- residual decay across repeated projection sweeps;
- perturbation growth or contraction between nearby operator trajectories;
- marginal sensitivity across SPN regions or evidence perturbations;
- path and branch comparisons after selecting readouts from an explicit DAG.

## Reported quantities

For signals (s_0,\ldots,s_L), the package reports

\[
n_\ell=\lVert s_\ell\rVert_2,\qquad
g_\ell=\frac{n_{\ell+1}}{n_\ell},\qquad
G_\ell=\frac{n_\ell}{n_0}.
\]

Zero-to-zero transfer is assigned gain one, while zero-to-positive transfer is amplification with
infinite gain. It also reports an empirical log-norm slope and fit quality. These are descriptive
statistics, not proofs.

## Three distinct conclusions

1. **Observed attenuation:** norms decreased in the recorded run.
2. **Observed monotone non-increase:** every measured local gain was at most one within tolerance.
3. **Certified exponential attenuation:** independent analytic bounds
   (\lVert s_{\ell+1}\rVert\leq \rho_\ell\lVert s_\ell\rVert) were supplied and
   (\max_\ell\rho_\ell<1).

Only the third conclusion supports the envelope

\[
\lVert s_L\rVert\leq \rho^L\lVert s_0\rVert,
\qquad \rho=\max_\ell\rho_\ell<1.
\]

Strict decrease in a finite experiment does not by itself establish exponential attenuation; gains
may approach one. This distinction directly prevents the overclaim highlighted in the supplied
review discussion.

## Fair cross-paradigm protocol

For a controlled comparison:

1. use the same architecture, batch, initialization, and local relation;
2. record the channel and granularity explicitly;
3. retain raw signals as well as norms;
4. compare both local gains and cumulative transfer;
5. report repeated sweeps against wall-clock and memory measurements when performance is studied;
6. separate measured results from analytic contraction bounds;
7. keep optimizer transforms of projection residuals distinct from the original projection map;
8. report inconsistent or noisy constraint residuals rather than assuming feasibility.

`examples/signal_attenuation.py` demonstrates a certified cotangent contraction and an empirical
target trace that contains amplification and therefore receives no exponential certificate.

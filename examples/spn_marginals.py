# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Section 5: exact variable and gate marginals of a complete/decomposable SPN."""

from __future__ import annotations

import jax.numpy as jnp

from kl_projection_message_passing import (
    Indicator,
    Product,
    Sum,
    SumProductNetwork,
    enumerate_posterior,
)


def main() -> None:
    spn = SumProductNetwork(
        cardinalities=(2, 2),
        nodes=(
            Indicator(0, 0),
            Indicator(0, 1),
            Indicator(1, 0),
            Indicator(1, 1),
            Product((0, 2)),
            Product((1, 3)),
            Sum((4, 5), jnp.asarray([0.7, 0.3])),
        ),
        root=6,
    )
    evidence = (jnp.asarray([0.8, 0.2]), jnp.asarray([0.3, 0.9]))
    readout = spn.upward_downward(evidence)
    enumerated = enumerate_posterior(spn, evidence)

    print(f"evidence normalizer: {float(readout.root_value):.8f}")
    for variable, marginal in enumerate(readout.variable_marginals):
        error = jnp.max(jnp.abs(marginal - enumerated.variable_marginals[variable]))
        print(f"variable {variable} posterior: {marginal}; enumeration error: {float(error):.3e}")
    print(f"root gate posterior: {readout.gate_conditionals[spn.root]}")


if __name__ == "__main__":
    main()

# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Compare a finite nonlinear operator with its differential and projection corollary."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from kl_projection_message_passing import (
    ComposedOperator,
    OperatorStage,
    affine_projector,
    compare_finite_and_differential,
    euclidean_residual_corollary,
)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    normalized_positive_map = ComposedOperator(
        (
            OperatorStage("positive-kernel", jnp.exp),
            OperatorStage("normalization", lambda value: value / jnp.sum(value)),
        ),
        name="positive-normalization",
    )
    comparison = compare_finite_and_differential(
        normalized_positive_map,
        jnp.asarray([0.2, -0.4, 0.7]),
        jnp.asarray([0.5, -0.1, 0.2]),
        epsilons=(1e-2, 5e-3, 2.5e-3),
    )

    projector = affine_projector(jnp.asarray([[1.0, 1.0]]), jnp.asarray([1.0]))
    corollary = euclidean_residual_corollary(jnp.asarray([2.0, -0.5]), projector)
    report = {
        "cyclic_projection_corollary_error": float(corollary.cyclic_projection_error),
        "distance_gradient_identity_error": float(corollary.gradient_identity_error),
        "finite_differential_errors": [float(value) for value in comparison.absolute_errors],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""A finite conjugate projection update and its differential approximation."""

from __future__ import annotations

from kl_projection_message_passing import linear_gaussian_projection_update


def main() -> None:
    update = linear_gaussian_projection_update(
        prior_mean=0.25,
        prior_variance=0.7,
        slope=1.4,
        intercept=-0.2,
        observation=1.1,
        relation_variance=0.3,
        observation_variance=0.5,
    )
    print(f"posterior mean: {float(update.posterior_mean):.8f}")
    print(f"posterior variance: {float(update.posterior_variance):.8f}")
    print(f"differential approximation: {float(update.differential_approximation):.8f}")
    print(f"one-step discrepancy: {float(update.one_step_discrepancy):.8f}")
    print(f"projection contraction: {float(update.projection_contraction_factor):.8f}")


if __name__ == "__main__":
    main()

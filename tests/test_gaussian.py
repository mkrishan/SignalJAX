# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kl_projection_message_passing.gaussian import (
    linear_gaussian_projection_update,
    linear_gaussian_recentered_trajectory,
    softened_message_score,
)

PARAMETERS = {
    "slope": 1.4,
    "intercept": -0.2,
    "observation": 1.1,
    "relation_variance": 0.3,
    "observation_variance": 0.5,
}


def test_conjugate_update_matches_closed_form_posterior() -> None:
    mean = 0.25
    variance = 0.7
    update = linear_gaussian_projection_update(mean, variance, **PARAMETERS)
    effective = PARAMETERS["relation_variance"] + PARAMETERS["observation_variance"]
    expected_variance = 1.0 / (1.0 / variance + PARAMETERS["slope"] ** 2 / effective)
    expected_mean = expected_variance * (
        mean / variance
        + PARAMETERS["slope"] * (PARAMETERS["observation"] - PARAMETERS["intercept"]) / effective
    )
    np.testing.assert_allclose(float(update.posterior_variance), expected_variance, atol=1e-12)
    np.testing.assert_allclose(float(update.posterior_mean), expected_mean, atol=1e-12)


def test_message_score_matches_analytic_log_likelihood_derivative() -> None:
    mean = 0.25
    score = softened_message_score(mean, **PARAMETERS)
    effective = PARAMETERS["relation_variance"] + PARAMETERS["observation_variance"]
    residual = PARAMETERS["observation"] - (PARAMETERS["slope"] * mean + PARAMETERS["intercept"])
    expected = PARAMETERS["slope"] * residual / effective
    np.testing.assert_allclose(float(score), expected, atol=1e-12)


def test_reported_contraction_is_derivative_of_recentered_projection_map() -> None:
    variance = 0.7

    def projection_map(mean: jax.Array) -> jax.Array:
        return linear_gaussian_projection_update(mean, variance, **PARAMETERS).posterior_mean

    mean = jnp.asarray(0.25)
    update = linear_gaussian_projection_update(mean, variance, **PARAMETERS)
    derivative = jax.grad(projection_map)(mean)
    np.testing.assert_allclose(
        np.asarray(derivative), np.asarray(update.projection_contraction_factor), atol=1e-12
    )


def test_narrow_belief_reduces_finite_vs_differential_discrepancy() -> None:
    broad = linear_gaussian_projection_update(0.25, 0.5, **PARAMETERS)
    narrow = linear_gaussian_projection_update(0.25, 0.01, **PARAMETERS)
    assert abs(float(narrow.one_step_discrepancy)) < abs(float(broad.one_step_discrepancy))


def test_recentered_trajectories_follow_reported_contraction_factors() -> None:
    trajectory = linear_gaussian_recentered_trajectory(
        0.25,
        0.2,
        steps=4,
        **PARAMETERS,
    )
    projection_ratios = trajectory.projection_errors[1:] / trajectory.projection_errors[:-1]
    differential_ratios = trajectory.differential_errors[1:] / trajectory.differential_errors[:-1]
    np.testing.assert_allclose(
        np.asarray(projection_ratios), trajectory.projection_contraction_factor, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(differential_ratios), trajectory.differential_contraction_factor, atol=1e-12
    )

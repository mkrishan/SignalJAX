# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Scalar linear-Gaussian softening used as a finite projection example."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class LinearGaussianUpdate(NamedTuple):
    """Exact conjugate update and its local differential comparison."""

    posterior_mean: Array
    posterior_variance: Array
    effective_noise_variance: Array
    log_likelihood_score: Array
    differential_approximation: Array
    one_step_discrepancy: Array
    projection_contraction_factor: Array
    differential_contraction_factor: Array


class LinearGaussianTrajectory(NamedTuple):
    """Fixed-variance recentering trajectories for finite and differential updates."""

    optimum: Array
    projection_means: Array
    differential_means: Array
    projection_errors: Array
    differential_errors: Array
    projection_contraction_factor: Array
    differential_contraction_factor: Array


def softened_observation_log_message(
    parameter: ArrayLike,
    *,
    slope: ArrayLike,
    intercept: ArrayLike = 0.0,
    observation: ArrayLike,
    relation_variance: ArrayLike,
    observation_variance: ArrayLike,
) -> Array:
    """Log message obtained after marginalizing a softened linear relation.

    The model is ``y | parameter ~ Normal(slope * parameter + intercept, relation_variance)``
    followed by ``observation | y ~ Normal(y, observation_variance)``.
    """

    parameter = jnp.asarray(parameter)
    slope = jnp.asarray(slope)
    intercept = jnp.asarray(intercept)
    observation = jnp.asarray(observation)
    variance = jnp.asarray(relation_variance) + jnp.asarray(observation_variance)
    residual = observation - (slope * parameter + intercept)
    return -0.5 * (jnp.log(2 * jnp.pi * variance) + residual**2 / variance)


def softened_message_score(
    parameter: ArrayLike,
    *,
    slope: ArrayLike,
    intercept: ArrayLike = 0.0,
    observation: ArrayLike,
    relation_variance: ArrayLike,
    observation_variance: ArrayLike,
) -> Array:
    """Differentiate the marginalized positive message in its parameter."""

    return jax.grad(softened_observation_log_message)(
        jnp.asarray(parameter),
        slope=slope,
        intercept=intercept,
        observation=observation,
        relation_variance=relation_variance,
        observation_variance=observation_variance,
    )


def linear_gaussian_projection_update(
    prior_mean: ArrayLike,
    prior_variance: ArrayLike,
    *,
    slope: ArrayLike,
    intercept: ArrayLike = 0.0,
    observation: ArrayLike,
    relation_variance: ArrayLike,
    observation_variance: ArrayLike,
) -> LinearGaussianUpdate:
    """Return the exact scalar posterior update and differential approximation.

    The exact posterior mean is a one-sweep sum-product/KL-projection readout on the conjugate tree.
    The comparison uses ``prior_variance * score`` as the narrow-belief differential step. It is a
    local worked example, not a claim that general projection and gradient trajectories coincide.
    """

    mean = jnp.asarray(prior_mean)
    variance = jnp.asarray(prior_variance)
    slope = jnp.asarray(slope)
    effective_noise = jnp.asarray(relation_variance) + jnp.asarray(observation_variance)
    score = softened_message_score(
        mean,
        slope=slope,
        intercept=intercept,
        observation=observation,
        relation_variance=relation_variance,
        observation_variance=observation_variance,
    )
    posterior_variance = 1.0 / (1.0 / variance + slope**2 / effective_noise)
    posterior_mean = mean + posterior_variance * score
    differential_approximation = mean + variance * score
    projection_contraction = effective_noise / (effective_noise + variance * slope**2)
    differential_contraction = 1.0 - variance * slope**2 / effective_noise
    return LinearGaussianUpdate(
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        effective_noise_variance=effective_noise,
        log_likelihood_score=score,
        differential_approximation=differential_approximation,
        one_step_discrepancy=posterior_mean - differential_approximation,
        projection_contraction_factor=projection_contraction,
        differential_contraction_factor=differential_contraction,
    )


def linear_gaussian_recentered_trajectory(
    initial_mean: ArrayLike,
    fixed_variance: ArrayLike,
    *,
    slope: ArrayLike,
    intercept: ArrayLike = 0.0,
    observation: ArrayLike,
    relation_variance: ArrayLike,
    observation_variance: ArrayLike,
    steps: int,
) -> LinearGaussianTrajectory:
    """Compare repeated finite projection and differential recentering maps.

    After each conjugate sweep, the finite branch recenters a Gaussian with the original fixed
    variance at the posterior mean.  The differential branch takes ``variance * score`` and
    performs the same recentering.  This is the explicit toy learning rule discussed in the
    rebuttal; it is not part of the KL projection operator itself and does not imply trajectory
    equivalence elsewhere.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    slope_array = jnp.asarray(slope)
    if slope_array.ndim != 0 or float(slope_array) == 0.0:
        raise ValueError("the scalar trajectory requires a nonzero scalar slope")
    optimum = (jnp.asarray(observation) - jnp.asarray(intercept)) / slope_array
    projection_means = [jnp.asarray(initial_mean)]
    differential_means = [jnp.asarray(initial_mean)]

    for _ in range(steps):
        projection_update = linear_gaussian_projection_update(
            projection_means[-1],
            fixed_variance,
            slope=slope_array,
            intercept=intercept,
            observation=observation,
            relation_variance=relation_variance,
            observation_variance=observation_variance,
        )
        differential_update = linear_gaussian_projection_update(
            differential_means[-1],
            fixed_variance,
            slope=slope_array,
            intercept=intercept,
            observation=observation,
            relation_variance=relation_variance,
            observation_variance=observation_variance,
        )
        projection_means.append(projection_update.posterior_mean)
        differential_means.append(differential_update.differential_approximation)

    projection_array = jnp.stack(projection_means)
    differential_array = jnp.stack(differential_means)
    final_update = linear_gaussian_projection_update(
        initial_mean,
        fixed_variance,
        slope=slope_array,
        intercept=intercept,
        observation=observation,
        relation_variance=relation_variance,
        observation_variance=observation_variance,
    )
    return LinearGaussianTrajectory(
        optimum=optimum,
        projection_means=projection_array,
        differential_means=differential_array,
        projection_errors=projection_array - optimum,
        differential_errors=differential_array - optimum,
        projection_contraction_factor=final_update.projection_contraction_factor,
        differential_contraction_factor=final_update.differential_contraction_factor,
    )

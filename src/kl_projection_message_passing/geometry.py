# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Finite KL geometry used by the consensus/product operator.

The functions in this module are deliberately small and transformation-friendly. They perform
static shape checks but leave positivity checks to callers so they can be used under ``jax.jit``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class ProjectionResult(NamedTuple):
    """Intermediate and final states of a consensus/product projection."""

    consensus: Array
    factorized: Array
    consensus_residual: Array
    factorization_residual: Array


def normalize(weights: ArrayLike, axis: int | tuple[int, ...] | None = None) -> Array:
    """Normalize nonnegative weights along ``axis``.

    Inputs with zero total mass intentionally produce non-finite values: such inputs are outside the
    positive-simplex assumptions used by the paper.
    """

    values = jnp.asarray(weights)
    return values / jnp.sum(values, axis=axis, keepdims=axis is not None)


def normalize_log_weights(
    log_weights: ArrayLike, axis: int | tuple[int, ...] | None = None
) -> Array:
    """Return normalized log weights using a stable log-sum-exp calculation."""

    values = jnp.asarray(log_weights)
    return values - jax.scipy.special.logsumexp(values, axis=axis, keepdims=axis is not None)


def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    """Compute ``KL(p || q)`` with the standard zero-mass convention.

    ``p`` and ``q`` must be nonnegative arrays with the same shape. The function does not silently
    renormalize them because doing so could conceal a broken projection invariant.
    """

    p_array = jnp.asarray(p)
    q_array = jnp.asarray(q)
    if p_array.shape != q_array.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p_array.shape} and {q_array.shape}"
        )
    terms = xlogy(p_array, p_array) - xlogy(p_array, q_array)
    return jnp.sum(terms, axis=axis)


def diagonal_i_projection(joint: ArrayLike) -> Array:
    """I-project a replicated joint table onto its diagonal consensus face.

    Every axis represents one copy of the same finite variable. The result is supported only on
    entries ``(x, ..., x)`` and is the normalized restriction of ``joint`` to that diagonal.
    """

    table = jnp.asarray(joint)
    if table.ndim < 2:
        raise ValueError("a diagonal consensus projection needs at least two replicas")
    if len(set(table.shape)) != 1:
        raise ValueError(f"all replica alphabets must have equal size, got {table.shape}")

    states = jnp.arange(table.shape[0])
    diagonal_index = (states,) * table.ndim
    diagonal_mass = table[diagonal_index]
    diagonal_probability = normalize(diagonal_mass)
    return jnp.zeros_like(table).at[diagonal_index].set(diagonal_probability)


def coordinate_marginal(joint: ArrayLike, coordinate: int) -> Array:
    """Return one normalized coordinate marginal of a joint table."""

    table = normalize(jnp.asarray(joint))
    if not 0 <= coordinate < table.ndim:
        raise ValueError(f"coordinate {coordinate} is invalid for rank-{table.ndim} table")
    reduced_axes = tuple(axis for axis in range(table.ndim) if axis != coordinate)
    return jnp.sum(table, axis=reduced_axes) if reduced_axes else table


def product_reverse_kl_projection(joint: ArrayLike) -> Array:
    """Reverse-KL project a joint table onto the fully factorized product family.

    The minimizer is the outer product of the normalized table's coordinate marginals.
    """

    table = jnp.asarray(joint)
    if table.ndim < 1:
        raise ValueError("a product projection needs at least one coordinate")
    table = normalize(table)
    result = jnp.ones(table.shape, dtype=table.dtype)
    for coordinate in range(table.ndim):
        marginal = coordinate_marginal(table, coordinate)
        broadcast_shape = [1] * table.ndim
        broadcast_shape[coordinate] = table.shape[coordinate]
        result = result * jnp.reshape(marginal, broadcast_shape)
    return result


def consensus_product_operator(joint: ArrayLike) -> ProjectionResult:
    """Apply diagonal agreement followed by product restoration.

    This is the smallest finite-table realization of the paper's operator. Larger region graphs use
    the same two local operations with a replication schedule.
    """

    table = normalize(jnp.asarray(joint))
    consensus = diagonal_i_projection(table)
    factorized = product_reverse_kl_projection(consensus)
    return ProjectionResult(
        consensus=consensus,
        factorized=factorized,
        consensus_residual=kl_divergence(consensus, table),
        factorization_residual=kl_divergence(consensus, factorized),
    )


def pythagorean_gap(reference: ArrayLike, original: ArrayLike, projection: ArrayLike) -> Array:
    """Return the numerical gap in the primal KL Pythagorean identity."""

    return (
        kl_divergence(reference, original)
        - kl_divergence(reference, projection)
        - kl_divergence(projection, original)
    )

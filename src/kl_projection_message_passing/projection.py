# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Finite projection dynamics and their Euclidean residual corollary.

These routines are independent implementations of standard projection constructions.  They provide
the finite-update side of the paper's comparison without importing any external projection-learning
framework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .operators import PyTree, tree_add, tree_l2_norm, tree_scale, tree_subtract

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
Projector = Callable[[PyTree], PyTree]


@dataclass(frozen=True)
class ProjectionStage:
    """One named projection or constraint-restoration map."""

    name: str
    projector: Projector
    geometry: str = "euclidean"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("projection stage names must be non-empty")
        if not callable(self.projector):
            raise TypeError(f"projection stage {self.name!r} is not callable")
        if not self.geometry:
            raise ValueError("projection geometry labels must be non-empty")


class ProjectionTrajectory(NamedTuple):
    """Local states and residual norms from cyclic projection sweeps."""

    states: tuple[PyTree, ...]
    stage_names: tuple[str, ...]
    residual_norms: Array
    step_norms: Array
    sweeps: int
    relaxation: Array


def projection_residual(state: PyTree, projector: Projector) -> PyTree:
    """Return the correction ``projector(state) - state``."""

    return tree_subtract(projector(state), state)


def relaxed_projection(
    state: PyTree, projector: Projector, *, relaxation: ArrayLike = 1.0
) -> PyTree:
    """Apply a relaxed projection ``state + relaxation * (P(state) - state)``."""

    rate = jnp.asarray(relaxation)
    return tree_add(state, tree_scale(rate, projection_residual(state, projector)))


def cyclic_projections(
    stages: Sequence[ProjectionStage],
    initial_state: PyTree,
    *,
    sweeps: int,
    relaxation: ArrayLike = 1.0,
) -> ProjectionTrajectory:
    """Apply named local projectors cyclically and retain every local update."""

    stage_tuple = tuple(stages)
    if not stage_tuple:
        raise ValueError("cyclic projections need at least one stage")
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    rate = jnp.asarray(relaxation)
    if rate.ndim != 0 or not 0.0 < float(rate) <= 2.0:
        raise ValueError("relaxation must be a scalar in (0, 2]")

    states = [initial_state]
    names = ["initial"]
    residual_norms: list[Array] = []
    step_norms: list[Array] = []
    current = initial_state
    for sweep in range(sweeps):
        for stage in stage_tuple:
            residual = projection_residual(current, stage.projector)
            update = tree_scale(rate, residual)
            current = tree_add(current, update)
            residual_norms.append(tree_l2_norm(residual))
            step_norms.append(tree_l2_norm(update))
            states.append(current)
            names.append(f"sweep-{sweep + 1}:{stage.name}")
    return ProjectionTrajectory(
        states=tuple(states),
        stage_names=tuple(names),
        residual_norms=jnp.stack(residual_norms),
        step_norms=jnp.stack(step_norms),
        sweeps=sweeps,
        relaxation=rate,
    )


def affine_projector(matrix: ArrayLike, right_hand_side: ArrayLike) -> Callable[[ArrayLike], Array]:
    """Build the Euclidean projector onto ``{x: matrix @ x = right_hand_side}``.

    A pseudoinverse handles redundant affine constraints.  The matrix and right-hand side are fixed;
    projected values may still be transformed by ``jax.jit`` or differentiated.
    """

    operator = jnp.asarray(matrix)
    target = jnp.asarray(right_hand_side)
    if operator.ndim != 2:
        raise ValueError("an affine constraint matrix must have rank two")
    if target.shape != (operator.shape[0],):
        raise ValueError(
            f"right-hand side has shape {target.shape}; expected {(operator.shape[0],)}"
        )
    gram_pseudoinverse = jnp.linalg.pinv(operator @ operator.T)

    def project(value: ArrayLike) -> Array:
        point = jnp.asarray(value)
        if point.shape != (operator.shape[1],):
            raise ValueError(f"point has shape {point.shape}; expected {(operator.shape[1],)}")
        return point - operator.T @ (gram_pseudoinverse @ (operator @ point - target))

    return project


def box_projector(lower: ArrayLike, upper: ArrayLike) -> Callable[[ArrayLike], Array]:
    """Build the componentwise Euclidean projector onto a closed box."""

    lower_bound = jnp.asarray(lower)
    upper_bound = jnp.asarray(upper)
    if lower_bound.shape != upper_bound.shape:
        raise ValueError("box bounds must have the same shape")
    if bool(jnp.any(lower_bound > upper_bound)):
        raise ValueError("box lower bounds must not exceed upper bounds")

    def project(value: ArrayLike) -> Array:
        point = jnp.asarray(value)
        if point.shape != lower_bound.shape:
            raise ValueError(f"point has shape {point.shape}; expected {lower_bound.shape}")
        return jnp.clip(point, lower_bound, upper_bound)

    return project


def consensus_projector(
    values: Sequence[ArrayLike], weights: ArrayLike | None = None
) -> tuple[Array, ...]:
    """Project equal-shaped copies onto their weighted Euclidean consensus diagonal."""

    copies = tuple(jnp.asarray(value) for value in values)
    if not copies:
        raise ValueError("consensus projection needs at least one copy")
    if any(copy.shape != copies[0].shape for copy in copies[1:]):
        raise ValueError("all consensus copies must have the same shape")
    if weights is None:
        coefficients = jnp.full((len(copies),), 1.0 / len(copies), dtype=copies[0].dtype)
    else:
        raw_weights = jnp.asarray(weights, dtype=copies[0].dtype)
        if raw_weights.shape != (len(copies),):
            raise ValueError(f"weights have shape {raw_weights.shape}; expected {(len(copies),)}")
        if bool(jnp.any(raw_weights <= 0)):
            raise ValueError("consensus weights must be positive")
        coefficients = raw_weights / jnp.sum(raw_weights)
    average = sum(
        (coefficient * copy for coefficient, copy in zip(coefficients, copies, strict=True)),
        jnp.zeros_like(copies[0]),
    )
    return tuple(jnp.asarray(average) for _ in copies)


def half_squared_projection_distance(
    state: ArrayLike, projector: Callable[[ArrayLike], Array]
) -> Array:
    """Return ``0.5 * ||state - projector(state)||^2`` for an array state."""

    point = jnp.asarray(state)
    residual = point - projector(point)
    return 0.5 * jnp.vdot(jnp.ravel(residual), jnp.ravel(residual)).real


class EuclideanResidualCorollary(NamedTuple):
    """Numerical check of the Euclidean projection-residual identities."""

    projected_state: Array
    residual_gradient: Array
    autodiff_distance_gradient: Array
    gradient_identity_error: Array
    unit_sgd_state: Array
    cyclic_projection_error: Array


def euclidean_residual_corollary(
    state: ArrayLike, projector: Callable[[ArrayLike], Array]
) -> EuclideanResidualCorollary:
    """Check when residual descent is exactly a Euclidean projection step.

    For a unique Euclidean projection onto a regular closed convex set,
    ``grad(0.5 * distance(x, C)**2) = x - P_C(x)``.  An SGD step of size one therefore gives
    ``P_C(x)``.  This generic identity is the operator-level corollary used to place cyclic
    projection training systems in the calculus; it does not cover arbitrary geometries or nonlinear
    transformations of the residual.
    """

    point = jnp.asarray(state)
    projected = projector(point)
    residual_gradient = point - projected
    autodiff_gradient = jax.grad(lambda value: half_squared_projection_distance(value, projector))(
        point
    )
    unit_sgd_state = point - residual_gradient
    return EuclideanResidualCorollary(
        projected_state=projected,
        residual_gradient=residual_gradient,
        autodiff_distance_gradient=autodiff_gradient,
        gradient_identity_error=jnp.linalg.norm(autodiff_gradient - residual_gradient),
        unit_sgd_state=unit_sgd_state,
        cyclic_projection_error=jnp.linalg.norm(unit_sgd_state - projected),
    )

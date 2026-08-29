# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Composable operators, finite iterations, and differential comparisons.

The central abstraction is deliberately smaller than a training framework.  A composed operator
acts on an arbitrary JAX pytree, while a readout assigns the signal whose propagation is being
studied.  This keeps the finite operator, its differential, and its semantic readout separate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
PyTree = Any
Transform = Callable[[PyTree], PyTree]
Readout = Callable[[PyTree], PyTree]


def tree_add(left: PyTree, right: PyTree) -> PyTree:
    """Add two pytrees leafwise."""

    return jax.tree_util.tree_map(jnp.add, left, right)


def tree_subtract(left: PyTree, right: PyTree) -> PyTree:
    """Subtract two pytrees leafwise."""

    return jax.tree_util.tree_map(jnp.subtract, left, right)


def tree_scale(scale: ArrayLike, tree: PyTree) -> PyTree:
    """Multiply every leaf in a pytree by one scalar."""

    scalar = jnp.asarray(scale)
    return jax.tree_util.tree_map(lambda leaf: scalar * jnp.asarray(leaf), tree)


def tree_l2_norm(tree: PyTree) -> Array:
    """Return the Euclidean norm across all leaves of a pytree."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("a signal pytree must contain at least one leaf")
    squared_norms = tuple(
        jnp.real(jnp.vdot(jnp.ravel(jnp.asarray(leaf)), jnp.ravel(jnp.asarray(leaf))))
        for leaf in leaves
    )
    return jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))


@dataclass(frozen=True)
class OperatorStage:
    """One named local map in a composed message-passing operator."""

    name: str
    transform: Transform
    role: str = "local"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operator stage names must be non-empty")
        if not callable(self.transform):
            raise TypeError(f"operator stage {self.name!r} is not callable")


class OperatorTrace(NamedTuple):
    """States before and after every stage of one composed evaluation."""

    states: tuple[PyTree, ...]
    stage_names: tuple[str, ...]

    @property
    def final_state(self) -> PyTree:
        """Return the state after the last stage."""

        return self.states[-1]


@dataclass(frozen=True)
class ComposedOperator:
    """An ordered composition of local maps on a common lifted state."""

    stages: tuple[OperatorStage, ...]
    name: str = "composed-operator"

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a composed operator needs at least one stage")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("operator stage names must be unique")
        if not self.name:
            raise ValueError("operator names must be non-empty")

    def __call__(self, state: PyTree) -> PyTree:
        """Apply the complete composition without retaining intermediate states."""

        current = state
        for stage in self.stages:
            current = stage.transform(current)
        return current

    def trace(self, state: PyTree) -> OperatorTrace:
        """Apply the composition and retain every intermediate state."""

        states = [state]
        current = state
        for stage in self.stages:
            current = stage.transform(current)
            states.append(current)
        return OperatorTrace(
            states=tuple(states),
            stage_names=("input", *(stage.name for stage in self.stages)),
        )

    def differential(self, state: PyTree, tangent: PyTree) -> PyTree:
        """Apply the JAX differential of the complete operator to ``tangent``."""

        return jax.jvp(self, (state,), (tangent,))[1]


class IterationTrace(NamedTuple):
    """Finite trajectory and residual norms of repeated operator evaluation."""

    states: tuple[PyTree, ...]
    fixed_point_residuals: Array
    step_norms: Array
    relaxation: Array


def relaxed_operator_step(
    operator: Transform, state: PyTree, relaxation: ArrayLike = 1.0
) -> PyTree:
    """Return ``state + relaxation * (operator(state) - state)``."""

    rate = jnp.asarray(relaxation)
    proposal = operator(state)
    return tree_add(state, tree_scale(rate, tree_subtract(proposal, state)))


def fixed_point_residual(operator: Transform, state: PyTree) -> Array:
    """Return ``||operator(state) - state||`` across the entire state pytree."""

    return tree_l2_norm(tree_subtract(operator(state), state))


def iterate_operator(
    operator: Transform,
    initial_state: PyTree,
    *,
    steps: int,
    relaxation: ArrayLike = 1.0,
) -> IterationTrace:
    """Iterate a finite operator and retain diagnostic residuals.

    The function reports what the finite dynamics did.  It makes no convergence claim unless one is
    established independently for the supplied operator and state space.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    rate = jnp.asarray(relaxation)
    if rate.ndim != 0 or not 0.0 < float(rate) <= 2.0:
        raise ValueError("relaxation must be a scalar in (0, 2]")

    states = [initial_state]
    residuals: list[Array] = []
    step_norms: list[Array] = []
    current = initial_state
    for _ in range(steps):
        proposal = operator(current)
        residual = tree_subtract(proposal, current)
        update = tree_scale(rate, residual)
        current = tree_add(current, update)
        residuals.append(tree_l2_norm(residual))
        step_norms.append(tree_l2_norm(update))
        states.append(current)
    return IterationTrace(
        states=tuple(states),
        fixed_point_residuals=jnp.stack(residuals),
        step_norms=jnp.stack(step_norms),
        relaxation=rate,
    )


class DifferentialComparison(NamedTuple):
    """Central finite differences compared with one exact JAX JVP readout."""

    epsilons: Array
    differential: PyTree
    finite_differences: tuple[PyTree, ...]
    absolute_errors: Array
    relative_errors: Array


def compare_finite_and_differential(
    operator: Transform,
    state: PyTree,
    direction: PyTree,
    *,
    epsilons: Sequence[float],
    readout: Readout = lambda value: value,
) -> DifferentialComparison:
    """Compare finite changes of a readout with its local differential.

    This is the numerical bridge between iterating a finite map and evaluating its linearization.
    It does not imply that repeated finite updates and repeated differential updates share
    trajectories.
    """

    epsilon_values = jnp.asarray(tuple(epsilons))
    if epsilon_values.ndim != 1 or epsilon_values.size == 0:
        raise ValueError("epsilons must be a non-empty one-dimensional sequence")
    if bool(jnp.any(epsilon_values <= 0)):
        raise ValueError("all finite-difference scales must be positive")

    def mapped_readout(value: PyTree) -> PyTree:
        return readout(operator(value))

    _, differential = jax.jvp(mapped_readout, (state,), (direction,))
    differential_norm = tree_l2_norm(differential)
    finite_differences: list[PyTree] = []
    absolute_errors: list[Array] = []
    relative_errors: list[Array] = []
    for epsilon in epsilon_values:
        positive = tree_add(state, tree_scale(epsilon, direction))
        negative = tree_subtract(state, tree_scale(epsilon, direction))
        numerator = tree_subtract(mapped_readout(positive), mapped_readout(negative))
        finite_difference = tree_scale(1.0 / (2.0 * epsilon), numerator)
        error = tree_l2_norm(tree_subtract(finite_difference, differential))
        finite_differences.append(finite_difference)
        absolute_errors.append(error)
        relative_errors.append(error / jnp.maximum(differential_norm, jnp.finfo(error.dtype).eps))

    return DifferentialComparison(
        epsilons=epsilon_values,
        differential=differential,
        finite_differences=tuple(finite_differences),
        absolute_errors=jnp.stack(absolute_errors),
        relative_errors=jnp.stack(relative_errors),
    )

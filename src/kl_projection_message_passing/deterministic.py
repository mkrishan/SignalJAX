# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Deterministic computation lifts and logarithmic differential readouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
Primitive = Callable[..., ArrayLike]


@dataclass(frozen=True)
class Operation:
    """One local assignment ``name = function(*parents)`` in a computation DAG."""

    name: str
    parents: tuple[str, ...]
    function: Primitive

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operation names must be non-empty")
        if not self.parents:
            raise ValueError(f"operation {self.name!r} must have at least one parent")
        if not callable(self.function):
            raise TypeError(f"operation {self.name!r} does not have a callable function")


class DeterministicReadout(NamedTuple):
    """Forward values and path-accumulated reverse readouts."""

    values: dict[str, Array]
    adjoints: dict[str, Array]
    output_seed: Array


@dataclass(frozen=True)
class DeterministicGraph:
    """A small explicit DAG whose local pullbacks realize the paper's readout recursion.

    Operations must be in topological order. Node values are JAX arrays; branching is allowed and
    reverse contributions from all downstream paths are added at the shared parent.
    """

    inputs: tuple[str, ...]
    operations: tuple[Operation, ...]
    output: str

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("a deterministic graph needs at least one input")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("input names must be unique")

        available = set(self.inputs)
        for operation in self.operations:
            if operation.name in available:
                raise ValueError(f"duplicate node name {operation.name!r}")
            missing = set(operation.parents) - available
            if missing:
                raise ValueError(
                    f"operation {operation.name!r} appears before parents {sorted(missing)!r}"
                )
            available.add(operation.name)
        if self.output not in available:
            raise ValueError(f"unknown output node {self.output!r}")

    def _coerce_inputs(self, input_values: Mapping[str, ArrayLike]) -> dict[str, Array]:
        missing = set(self.inputs) - set(input_values)
        extra = set(input_values) - set(self.inputs)
        if missing or extra:
            raise ValueError(f"input mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        return {name: jnp.asarray(input_values[name]) for name in self.inputs}

    def forward(self, input_values: Mapping[str, ArrayLike]) -> dict[str, Array]:
        """Evaluate every node in topological order."""

        values = self._coerce_inputs(input_values)
        for operation in self.operations:
            arguments = tuple(values[parent] for parent in operation.parents)
            values[operation.name] = jnp.asarray(operation.function(*arguments))
        return values

    def readout(
        self,
        input_values: Mapping[str, ArrayLike],
        *,
        output_cotangent: ArrayLike | None = None,
        log_potential: Callable[[Array], ArrayLike] | None = None,
    ) -> DeterministicReadout:
        """Compute local logarithmic readouts by an explicit reverse traversal.

        Supply either an output cotangent, or a scalar ``log_potential`` whose gradient supplies the
        seed ``grad(log(phi))(output)``. With neither argument, an all-ones cotangent is used.
        """

        if output_cotangent is not None and log_potential is not None:
            raise ValueError("provide output_cotangent or log_potential, not both")

        values = self._coerce_inputs(input_values)
        pullbacks: dict[str, Callable[[Array], tuple[Array, ...]]] = {}
        for operation in self.operations:
            arguments = tuple(values[parent] for parent in operation.parents)
            value, pullback = jax.vjp(operation.function, *arguments)
            values[operation.name] = jnp.asarray(value)
            pullbacks[operation.name] = pullback

        output_value = values[self.output]
        if log_potential is not None:
            seed = jax.grad(lambda value: jnp.asarray(log_potential(value)))(output_value)
        elif output_cotangent is None:
            seed = jnp.ones_like(output_value)
        else:
            seed = jnp.asarray(output_cotangent)
            if seed.shape != output_value.shape:
                raise ValueError(
                    f"output cotangent has shape {seed.shape}; expected {output_value.shape}"
                )

        adjoints = {name: jnp.zeros_like(value) for name, value in values.items()}
        adjoints[self.output] = seed
        for operation in reversed(self.operations):
            contributions = pullbacks[operation.name](adjoints[operation.name])
            for parent, contribution in zip(operation.parents, contributions, strict=True):
                adjoints[parent] = adjoints[parent] + contribution

        return DeterministicReadout(values=values, adjoints=adjoints, output_seed=seed)


def deterministic_factor_message(
    function: Primitive,
    parent_values: Sequence[ArrayLike],
    parent_index: int,
    downstream_message: Callable[[Array], ArrayLike],
) -> Callable[[ArrayLike], Array]:
    """Build the local factor-to-parent message for a deterministic assignment.

    All parents except ``parent_index`` remain fixed at their forward values. This is the regular
    function represented by integrating a downstream message against a Dirac factor.
    """

    fixed_values = tuple(jnp.asarray(value) for value in parent_values)
    if not 0 <= parent_index < len(fixed_values):
        raise ValueError(f"parent_index {parent_index} is invalid for {len(fixed_values)} parents")

    def message(parent_value: ArrayLike) -> Array:
        arguments = list(fixed_values)
        arguments[parent_index] = jnp.asarray(parent_value)
        return jnp.asarray(downstream_message(jnp.asarray(function(*arguments))))

    return message


def log_message_differential(message: Callable[[Array], ArrayLike], point: ArrayLike) -> Array:
    """Differentiate the logarithm of a positive scalar message at ``point``."""

    return jax.grad(lambda value: jnp.log(jnp.asarray(message(value))))(jnp.asarray(point))


def log_message_directional_derivative(
    message: Callable[[Array], ArrayLike], point: ArrayLike, tangent: ArrayLike
) -> Array:
    """Evaluate a directional differential of a positive message's logarithm."""

    _, derivative = jax.jvp(
        lambda value: jnp.log(jnp.asarray(message(value))),
        (jnp.asarray(point),),
        (jnp.asarray(tangent),),
    )
    return derivative


def central_log_message_difference(
    message: Callable[[Array], ArrayLike],
    point: ArrayLike,
    direction: ArrayLike,
    step: ArrayLike,
) -> Array:
    """Central finite difference for checking a logarithmic directional readout."""

    x = jnp.asarray(point)
    v = jnp.asarray(direction)
    epsilon = jnp.asarray(step)
    return (
        jnp.log(jnp.asarray(message(x + epsilon * v)))
        - jnp.log(jnp.asarray(message(x - epsilon * v)))
    ) / (2 * epsilon)

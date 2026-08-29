# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Normalized sum-product messages for finite positive factor graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import normalize_log_weights

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
Node = tuple[str, int]
MessageKey = tuple[Node, Node]


@dataclass(frozen=True)
class Factor:
    """A strictly positive factor table with axes ordered as ``variables``."""

    variables: tuple[int, ...]
    potential: ArrayLike
    name: str = ""

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("factors must involve at least one variable")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"factor scope contains duplicates: {self.variables}")
        host_potential = np.asarray(self.potential)
        if not np.issubdtype(host_potential.dtype, np.number):
            raise TypeError("factor potentials must be numeric")
        if not np.all(np.isfinite(host_potential)) or not np.all(host_potential > 0):
            raise ValueError("factor potentials must be finite and strictly positive")
        object.__setattr__(self, "potential", jnp.asarray(self.potential))


@dataclass(frozen=True)
class DiscreteFactorGraph:
    """A finite factor graph with integer-indexed variables."""

    cardinalities: tuple[int, ...]
    factors: tuple[Factor, ...]

    def __post_init__(self) -> None:
        if not self.cardinalities or any(cardinality < 1 for cardinality in self.cardinalities):
            raise ValueError("all variable cardinalities must be positive")
        if not self.factors:
            raise ValueError("a factor graph needs at least one factor")
        for factor_index, factor in enumerate(self.factors):
            if any(
                variable < 0 or variable >= len(self.cardinalities) for variable in factor.variables
            ):
                raise ValueError(f"factor {factor_index} references an unknown variable")
            expected_shape = tuple(self.cardinalities[variable] for variable in factor.variables)
            if factor.potential.shape != expected_shape:
                raise ValueError(
                    f"factor {factor_index} has shape {factor.potential.shape}; "
                    f"expected {expected_shape} for scope {factor.variables}"
                )


class BeliefPropagationResult(NamedTuple):
    """Normalized beliefs and directed log messages from a BP run."""

    variable_beliefs: tuple[Array, ...]
    factor_beliefs: tuple[Array, ...]
    log_messages: dict[MessageKey, Array]
    converged: bool
    iterations: int
    max_message_delta: Array


class EnumerationResult(NamedTuple):
    """Exact normalized joint distribution and its marginals."""

    joint: Array
    variable_marginals: tuple[Array, ...]
    factor_marginals: tuple[Array, ...]
    partition: Array


def _variable_node(variable: int) -> Node:
    return ("variable", variable)


def _factor_node(factor: int) -> Node:
    return ("factor", factor)


def _adjacency(graph: DiscreteFactorGraph) -> dict[Node, list[Node]]:
    adjacency: dict[Node, list[Node]] = {
        _variable_node(variable): [] for variable in range(len(graph.cardinalities))
    }
    adjacency.update({_factor_node(index): [] for index in range(len(graph.factors))})
    for factor_index, factor in enumerate(graph.factors):
        factor_node = _factor_node(factor_index)
        for variable in factor.variables:
            variable_node = _variable_node(variable)
            adjacency[factor_node].append(variable_node)
            adjacency[variable_node].append(factor_node)
    return adjacency


def is_tree(graph: DiscreteFactorGraph) -> bool:
    """Return whether the entire bipartite factor graph is one connected tree."""

    adjacency = _adjacency(graph)
    if any(not neighbors for neighbors in adjacency.values()):
        return False
    start = next(iter(adjacency))
    visited: set[Node] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(neighbor for neighbor in adjacency[node] if neighbor not in visited)
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return len(visited) == len(adjacency) and edge_count == len(adjacency) - 1


def _tree_order(
    graph: DiscreteFactorGraph, root_variable: int
) -> tuple[list[Node], dict[Node, Node]]:
    if not is_tree(graph):
        raise ValueError("an exact upward-downward pass requires a connected tree factor graph")
    if not 0 <= root_variable < len(graph.cardinalities):
        raise ValueError(f"unknown root variable {root_variable}")

    adjacency = _adjacency(graph)
    root = _variable_node(root_variable)
    order: list[Node] = []
    parent: dict[Node, Node] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in reversed(adjacency[node]):
            if neighbor == parent.get(node):
                continue
            parent[neighbor] = node
            stack.append(neighbor)
    return order, parent


def _variable_to_factor(
    graph: DiscreteFactorGraph,
    variable: int,
    destination_factor: int,
    messages: dict[MessageKey, Array],
) -> Array:
    adjacency = _adjacency(graph)
    variable_node = _variable_node(variable)
    destination = _factor_node(destination_factor)
    log_message = jnp.zeros((graph.cardinalities[variable],))
    for neighbor in adjacency[variable_node]:
        if neighbor != destination:
            log_message = log_message + messages[(neighbor, variable_node)]
    return normalize_log_weights(log_message)


def _factor_to_variable(
    graph: DiscreteFactorGraph,
    factor_index: int,
    destination_variable: int,
    messages: dict[MessageKey, Array],
) -> Array:
    factor = graph.factors[factor_index]
    log_joint = jnp.log(jnp.asarray(factor.potential))
    destination_axis = factor.variables.index(destination_variable)
    factor_node = _factor_node(factor_index)

    for axis, variable in enumerate(factor.variables):
        if variable == destination_variable:
            continue
        variable_node = _variable_node(variable)
        incoming = messages[(variable_node, factor_node)]
        shape = [1] * len(factor.variables)
        shape[axis] = graph.cardinalities[variable]
        log_joint = log_joint + jnp.reshape(incoming, shape)

    reduced_axes = tuple(axis for axis in range(log_joint.ndim) if axis != destination_axis)
    if reduced_axes:
        log_message = jax.scipy.special.logsumexp(log_joint, axis=reduced_axes)
    else:
        log_message = log_joint
    return normalize_log_weights(log_message)


def _compute_message(
    graph: DiscreteFactorGraph,
    source: Node,
    destination: Node,
    messages: dict[MessageKey, Array],
) -> Array:
    if source[0] == "variable" and destination[0] == "factor":
        return _variable_to_factor(graph, source[1], destination[1], messages)
    if source[0] == "factor" and destination[0] == "variable":
        return _factor_to_variable(graph, source[1], destination[1], messages)
    raise ValueError(f"invalid factor-graph edge {source!r} -> {destination!r}")


def _beliefs_from_messages(
    graph: DiscreteFactorGraph, messages: dict[MessageKey, Array]
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    adjacency = _adjacency(graph)
    variable_beliefs: list[Array] = []
    for variable, cardinality in enumerate(graph.cardinalities):
        variable_node = _variable_node(variable)
        log_belief = jnp.zeros((cardinality,))
        for factor_node in adjacency[variable_node]:
            log_belief = log_belief + messages[(factor_node, variable_node)]
        variable_beliefs.append(jnp.exp(normalize_log_weights(log_belief)))

    factor_beliefs: list[Array] = []
    for factor_index, factor in enumerate(graph.factors):
        factor_node = _factor_node(factor_index)
        log_belief = jnp.log(jnp.asarray(factor.potential))
        for axis, variable in enumerate(factor.variables):
            incoming = messages[(_variable_node(variable), factor_node)]
            shape = [1] * len(factor.variables)
            shape[axis] = graph.cardinalities[variable]
            log_belief = log_belief + jnp.reshape(incoming, shape)
        factor_beliefs.append(jnp.exp(normalize_log_weights(log_belief)))
    return tuple(variable_beliefs), tuple(factor_beliefs)


def exact_tree_sum_product(
    graph: DiscreteFactorGraph, *, root_variable: int = 0
) -> BeliefPropagationResult:
    """Run one exact upward-downward normalized sum-product pass on a tree."""

    order, parent = _tree_order(graph, root_variable)
    messages: dict[MessageKey, Array] = {}

    for node in reversed(order[1:]):
        destination = parent[node]
        messages[(node, destination)] = _compute_message(graph, node, destination, messages)

    for node in order[1:]:
        source = parent[node]
        messages[(source, node)] = _compute_message(graph, source, node, messages)

    variable_beliefs, factor_beliefs = _beliefs_from_messages(graph, messages)
    return BeliefPropagationResult(
        variable_beliefs=variable_beliefs,
        factor_beliefs=factor_beliefs,
        log_messages=messages,
        converged=True,
        iterations=1,
        max_message_delta=jnp.asarray(0.0),
    )


def loopy_sum_product(
    graph: DiscreteFactorGraph,
    *,
    iterations: int = 50,
    damping: float = 0.0,
    tolerance: float = 1e-7,
) -> BeliefPropagationResult:
    """Run synchronous normalized sum-product iterations on a general graph.

    This routine reports numerical convergence only; it makes no exactness or convergence guarantee
    for loopy graphs.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must lie in [0, 1)")

    adjacency = _adjacency(graph)
    messages: dict[MessageKey, Array] = {}
    for source, neighbors in adjacency.items():
        for destination in neighbors:
            variable = source[1] if source[0] == "variable" else destination[1]
            cardinality = graph.cardinalities[variable]
            messages[(source, destination)] = jnp.full(
                (cardinality,), -jnp.log(jnp.asarray(cardinality, dtype=jnp.float32))
            )

    converged = False
    max_delta = jnp.asarray(jnp.inf)
    completed = 0
    for iteration in range(1, iterations + 1):
        completed = iteration
        candidates = {
            edge: _compute_message(graph, edge[0], edge[1], messages) for edge in messages
        }
        updated: dict[MessageKey, Array] = {}
        deltas: list[Array] = []
        for edge, candidate in candidates.items():
            if damping:
                probability = (1.0 - damping) * jnp.exp(candidate) + damping * jnp.exp(
                    messages[edge]
                )
                candidate = normalize_log_weights(jnp.log(probability))
            updated[edge] = candidate
            deltas.append(jnp.max(jnp.abs(jnp.exp(candidate) - jnp.exp(messages[edge]))))
        messages = updated
        max_delta = jnp.max(jnp.stack(deltas))
        if float(max_delta) <= tolerance:
            converged = True
            break

    variable_beliefs, factor_beliefs = _beliefs_from_messages(graph, messages)
    return BeliefPropagationResult(
        variable_beliefs=variable_beliefs,
        factor_beliefs=factor_beliefs,
        log_messages=messages,
        converged=converged,
        iterations=completed,
        max_message_delta=max_delta,
    )


def enumerate_distribution(graph: DiscreteFactorGraph) -> EnumerationResult:
    """Form the exact normalized joint table; intended for small verification problems."""

    joint = jnp.ones(graph.cardinalities)
    for factor in graph.factors:
        ordered = sorted(enumerate(factor.variables), key=lambda item: item[1])
        axis_permutation = tuple(original_axis for original_axis, _ in ordered)
        sorted_variables = tuple(variable for _, variable in ordered)
        potential = jnp.asarray(factor.potential)
        if axis_permutation != tuple(range(len(axis_permutation))):
            potential = jnp.transpose(potential, axis_permutation)
        shape = [1] * len(graph.cardinalities)
        for variable in sorted_variables:
            shape[variable] = graph.cardinalities[variable]
        joint = joint * jnp.reshape(potential, shape)

    partition = jnp.sum(joint)
    normalized_joint = joint / partition
    variable_marginals = tuple(
        jnp.sum(
            normalized_joint,
            axis=tuple(axis for axis in range(len(graph.cardinalities)) if axis != variable),
        )
        for variable in range(len(graph.cardinalities))
    )

    factor_marginals: list[Array] = []
    for factor in graph.factors:
        sorted_variables = tuple(sorted(factor.variables))
        reduced_axes = tuple(
            axis for axis in range(len(graph.cardinalities)) if axis not in sorted_variables
        )
        marginal = (
            jnp.sum(normalized_joint, axis=reduced_axes) if reduced_axes else normalized_joint
        )
        if factor.variables != sorted_variables:
            permutation = tuple(sorted_variables.index(variable) for variable in factor.variables)
            marginal = jnp.transpose(marginal, permutation)
        factor_marginals.append(marginal)

    return EnumerationResult(
        joint=normalized_joint,
        variable_marginals=variable_marginals,
        factor_marginals=tuple(factor_marginals),
        partition=partition,
    )

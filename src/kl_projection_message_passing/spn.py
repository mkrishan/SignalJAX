# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Complete and decomposable sum-product networks with exact readouts."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


@dataclass(frozen=True)
class Indicator:
    """An indicator leaf for one state of a finite variable."""

    variable: int
    state: int


@dataclass(frozen=True)
class Sum:
    """A positive weighted sum over complete child scopes."""

    children: tuple[int, ...]
    weights: ArrayLike

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("sum nodes need at least one child")
        weights = np.asarray(self.weights)
        if weights.shape != (len(self.children),):
            raise ValueError(
                f"sum weights have shape {weights.shape}; expected {(len(self.children),)}"
            )
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
            raise ValueError("sum weights must be finite and strictly positive")
        object.__setattr__(self, "weights", jnp.asarray(self.weights))


@dataclass(frozen=True)
class Product:
    """A product over children with pairwise disjoint scopes."""

    children: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("product nodes need at least one child")


SPNNode: TypeAlias = Indicator | Sum | Product


class SPNReadout(NamedTuple):
    """Upward values, accumulated contexts, and exact posterior readouts."""

    upward: tuple[Array, ...]
    downward: tuple[Array, ...]
    root_value: Array
    variable_marginals: tuple[Array, ...]
    gate_conditionals: tuple[Array | None, ...]
    gate_joint_masses: tuple[Array | None, ...]


class SPNEnumeration(NamedTuple):
    """Exact assignment posterior formed by explicit enumeration."""

    assignments: Array
    probabilities: Array
    variable_marginals: tuple[Array, ...]
    partition: Array


@dataclass(frozen=True)
class SumProductNetwork:
    """A topologically ordered positive SPN DAG.

    Children must have lower indices than their parent. Completeness and decomposability are checked
    at construction; this prevents sensitivity values from being mislabeled as exact marginals.
    """

    cardinalities: tuple[int, ...]
    nodes: tuple[SPNNode, ...]
    root: int
    scopes: tuple[frozenset[int], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.cardinalities or any(cardinality < 1 for cardinality in self.cardinalities):
            raise ValueError("all variable cardinalities must be positive")
        if not self.nodes:
            raise ValueError("an SPN needs at least one node")
        if not 0 <= self.root < len(self.nodes):
            raise ValueError(f"invalid root index {self.root}")

        scopes: list[frozenset[int]] = []
        for index, node in enumerate(self.nodes):
            if isinstance(node, Indicator):
                if not 0 <= node.variable < len(self.cardinalities):
                    raise ValueError(
                        f"indicator {index} references unknown variable {node.variable}"
                    )
                if not 0 <= node.state < self.cardinalities[node.variable]:
                    raise ValueError(
                        f"indicator {index} state {node.state} is invalid for variable "
                        f"{node.variable}"
                    )
                scopes.append(frozenset((node.variable,)))
                continue

            if any(child < 0 or child >= index for child in node.children):
                raise ValueError(
                    f"node {index} must reference only earlier children, got {node.children}"
                )
            child_scopes = tuple(scopes[child] for child in node.children)
            if isinstance(node, Sum):
                expected = child_scopes[0]
                if any(scope != expected for scope in child_scopes[1:]):
                    raise ValueError(
                        f"sum node {index} is incomplete: child scopes are {child_scopes}"
                    )
                scopes.append(expected)
            else:
                accumulated: set[int] = set()
                for scope in child_scopes:
                    overlap = accumulated.intersection(scope)
                    if overlap:
                        raise ValueError(
                            f"product node {index} is not decomposable; overlapping variables: "
                            f"{sorted(overlap)}"
                        )
                    accumulated.update(scope)
                scopes.append(frozenset(accumulated))

        reachable: set[int] = set()
        stack = [self.root]
        while stack:
            node_index = stack.pop()
            if node_index in reachable:
                continue
            reachable.add(node_index)
            node = self.nodes[node_index]
            if isinstance(node, (Sum, Product)):
                stack.extend(node.children)
        if len(reachable) != len(self.nodes):
            unreachable = sorted(set(range(len(self.nodes))) - reachable)
            raise ValueError(f"SPN contains nodes unreachable from root {self.root}: {unreachable}")
        if scopes[self.root] != frozenset(range(len(self.cardinalities))):
            raise ValueError(
                f"root scope is {sorted(scopes[self.root])}; expected all variables "
                f"{list(range(len(self.cardinalities)))}"
            )
        object.__setattr__(self, "scopes", tuple(scopes))

    def _coerce_evidence(self, evidence: tuple[ArrayLike, ...]) -> tuple[Array, ...]:
        if len(evidence) != len(self.cardinalities):
            raise ValueError(
                f"received evidence for {len(evidence)} variables; "
                f"expected {len(self.cardinalities)}"
            )
        arrays = tuple(jnp.asarray(item) for item in evidence)
        for variable, (array, cardinality) in enumerate(
            zip(arrays, self.cardinalities, strict=True)
        ):
            if array.shape != (cardinality,):
                raise ValueError(
                    f"evidence for variable {variable} has shape {array.shape}; "
                    f"expected {(cardinality,)}"
                )
        return arrays

    def evaluate(self, evidence: tuple[ArrayLike, ...]) -> tuple[Array, ...]:
        """Compute all upward circuit values."""

        evidence_arrays = self._coerce_evidence(evidence)
        values: list[Array] = []
        for node in self.nodes:
            if isinstance(node, Indicator):
                values.append(evidence_arrays[node.variable][node.state])
            elif isinstance(node, Sum):
                children = jnp.stack(tuple(values[child] for child in node.children))
                values.append(jnp.sum(jnp.asarray(node.weights) * children))
            else:
                children = jnp.stack(tuple(values[child] for child in node.children))
                values.append(jnp.prod(children))
        return tuple(values)

    def upward_downward(self, evidence: tuple[ArrayLike, ...]) -> SPNReadout:
        """Compute exact SPN variable and gate readouts in one upward-downward pass."""

        evidence_arrays = self._coerce_evidence(evidence)
        upward = self.evaluate(evidence_arrays)
        root_value = upward[self.root]
        downward = [jnp.zeros_like(root_value) for _ in self.nodes]
        downward[self.root] = jnp.ones_like(root_value)

        for index in reversed(range(len(self.nodes))):
            node = self.nodes[index]
            context = downward[index]
            if isinstance(node, Sum):
                for child, weight in zip(node.children, jnp.asarray(node.weights), strict=True):
                    downward[child] = downward[child] + context * weight
            elif isinstance(node, Product):
                child_values = tuple(upward[child] for child in node.children)
                prefix = [jnp.ones_like(root_value)]
                for value in child_values:
                    prefix.append(prefix[-1] * value)
                suffix = [jnp.ones_like(root_value)] * (len(child_values) + 1)
                for child_position in reversed(range(len(child_values))):
                    suffix[child_position] = (
                        suffix[child_position + 1] * child_values[child_position]
                    )
                for child_position, child in enumerate(node.children):
                    sibling_product = prefix[child_position] * suffix[child_position + 1]
                    downward[child] = downward[child] + context * sibling_product

        variable_marginals = [
            jnp.zeros((cardinality,), dtype=root_value.dtype) for cardinality in self.cardinalities
        ]
        for index, node in enumerate(self.nodes):
            if isinstance(node, Indicator):
                posterior_mass = (
                    evidence_arrays[node.variable][node.state] * downward[index] / root_value
                )
                variable_marginals[node.variable] = (
                    variable_marginals[node.variable].at[node.state].add(posterior_mass)
                )

        gate_conditionals: list[Array | None] = [None] * len(self.nodes)
        gate_joint_masses: list[Array | None] = [None] * len(self.nodes)
        for index, node in enumerate(self.nodes):
            if isinstance(node, Sum):
                weighted_children = jnp.asarray(node.weights) * jnp.stack(
                    tuple(upward[child] for child in node.children)
                )
                gate_conditionals[index] = weighted_children / upward[index]
                gate_joint_masses[index] = downward[index] * weighted_children / root_value

        return SPNReadout(
            upward=upward,
            downward=tuple(downward),
            root_value=root_value,
            variable_marginals=tuple(variable_marginals),
            gate_conditionals=tuple(gate_conditionals),
            gate_joint_masses=tuple(gate_joint_masses),
        )


def log_evidence_gradient(
    spn: SumProductNetwork, log_evidence: tuple[ArrayLike, ...]
) -> tuple[Array, ...]:
    """Differentiate ``log S(e)`` with respect to every log-evidence table."""

    log_arrays = tuple(jnp.asarray(item) for item in log_evidence)

    def log_value(log_tables: tuple[Array, ...]) -> Array:
        values = spn.evaluate(tuple(jnp.exp(table) for table in log_tables))
        return jnp.log(values[spn.root])

    return jax.grad(log_value)(log_arrays)


def enumerate_posterior(spn: SumProductNetwork, evidence: tuple[ArrayLike, ...]) -> SPNEnumeration:
    """Enumerate the posterior represented by an SPN; intended for tests and tiny examples."""

    evidence_arrays = spn._coerce_evidence(evidence)
    assignment_tuples = tuple(itertools.product(*(range(card) for card in spn.cardinalities)))
    unnormalized: list[Array] = []
    for assignment in assignment_tuples:
        hard_evidence = tuple(
            jax.nn.one_hot(state, cardinality)
            for state, cardinality in zip(assignment, spn.cardinalities, strict=True)
        )
        base_mass = spn.evaluate(hard_evidence)[spn.root]
        likelihood = jnp.prod(
            jnp.stack(
                tuple(evidence_arrays[variable][state] for variable, state in enumerate(assignment))
            )
        )
        unnormalized.append(base_mass * likelihood)

    masses = jnp.stack(unnormalized)
    partition = jnp.sum(masses)
    probabilities = masses / partition
    assignments = jnp.asarray(assignment_tuples, dtype=jnp.int32)
    marginals = tuple(
        jnp.stack(
            tuple(
                jnp.sum(probabilities * (assignments[:, variable] == state))
                for state in range(cardinality)
            )
        )
        for variable, cardinality in enumerate(spn.cardinalities)
    )
    return SPNEnumeration(
        assignments=assignments,
        probabilities=probabilities,
        variable_marginals=marginals,
        partition=partition,
    )


def unfold_spn(spn: SumProductNetwork) -> tuple[SumProductNetwork, tuple[int, ...]]:
    """Unfold an SPN DAG into a tree and return the copy-to-original node map."""

    unfolded_nodes: list[SPNNode] = []
    origin: list[int] = []

    def clone(original_index: int) -> int:
        original = spn.nodes[original_index]
        if isinstance(original, Indicator):
            copied: SPNNode = Indicator(original.variable, original.state)
        else:
            copied_children = tuple(clone(child) for child in original.children)
            if isinstance(original, Sum):
                copied = Sum(copied_children, jnp.asarray(original.weights))
            else:
                copied = Product(copied_children)
        copied_index = len(unfolded_nodes)
        unfolded_nodes.append(copied)
        origin.append(original_index)
        return copied_index

    unfolded_root = clone(spn.root)
    unfolded = SumProductNetwork(
        cardinalities=spn.cardinalities,
        nodes=tuple(unfolded_nodes),
        root=unfolded_root,
    )
    return unfolded, tuple(origin)


def merge_unfolded_downward(
    original: SumProductNetwork,
    unfolded_readout: SPNReadout,
    origin: tuple[int, ...],
) -> tuple[Array, ...]:
    """Merge unfolded contexts by adding all copies of each original DAG node."""

    if len(origin) != len(unfolded_readout.downward):
        raise ValueError("origin map and unfolded readout have different sizes")
    merged = [jnp.zeros_like(unfolded_readout.root_value) for _ in original.nodes]
    for copied_index, original_index in enumerate(origin):
        if not 0 <= original_index < len(original.nodes):
            raise ValueError(f"origin map contains invalid node {original_index}")
        merged[original_index] = merged[original_index] + unfolded_readout.downward[copied_index]
    return tuple(merged)

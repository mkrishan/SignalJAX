# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kl_projection_message_passing.factor_graph import (
    DiscreteFactorGraph,
    Factor,
    enumerate_distribution,
    exact_tree_sum_product,
    is_tree,
    loopy_sum_product,
)


def binary_chain() -> DiscreteFactorGraph:
    return DiscreteFactorGraph(
        cardinalities=(2, 2, 2),
        factors=(
            Factor((0,), jnp.asarray([0.65, 0.35]), "left evidence"),
            Factor((0, 1), jnp.asarray([[2.0, 0.5], [0.4, 1.8]]), "left coupling"),
            Factor((1, 2), jnp.asarray([[1.7, 0.3], [0.6, 2.1]]), "right coupling"),
            Factor((2,), jnp.asarray([0.25, 0.75]), "right evidence"),
        ),
    )


def test_exact_tree_sum_product_matches_enumeration() -> None:
    graph = binary_chain()
    assert is_tree(graph)
    belief_propagation = exact_tree_sum_product(graph, root_variable=1)
    enumeration = enumerate_distribution(graph)

    for actual, expected in zip(
        belief_propagation.variable_beliefs, enumeration.variable_marginals, strict=True
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)
    for actual, expected in zip(
        belief_propagation.factor_beliefs, enumeration.factor_marginals, strict=True
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


def test_loopy_schedule_converges_to_tree_solution_on_a_tree() -> None:
    graph = binary_chain()
    exact = exact_tree_sum_product(graph)
    iterative = loopy_sum_product(graph, iterations=50, tolerance=1e-12)
    assert iterative.converged
    for actual, expected in zip(iterative.variable_beliefs, exact.variable_beliefs, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)


def test_cycle_is_not_mislabeled_as_an_exact_tree() -> None:
    graph = DiscreteFactorGraph(
        cardinalities=(2, 2),
        factors=(
            Factor((0, 1), jnp.ones((2, 2))),
            Factor((0, 1), jnp.asarray([[2.0, 1.0], [1.0, 2.0]])),
        ),
    )
    assert not is_tree(graph)
    with pytest.raises(ValueError, match="connected tree"):
        exact_tree_sum_product(graph)


def test_factor_shape_and_positivity_are_validated() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        Factor((0,), jnp.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="expected"):
        DiscreteFactorGraph((2,), (Factor((0,), jnp.ones((3,))),))


@given(st.lists(st.floats(min_value=0.05, max_value=4.0), min_size=10, max_size=10))
@settings(max_examples=12, deadline=None)
def test_random_positive_binary_chain_matches_enumeration(values: list[float]) -> None:
    graph = DiscreteFactorGraph(
        cardinalities=(2, 2),
        factors=(
            Factor((0,), jnp.asarray(values[:2])),
            Factor((0, 1), jnp.asarray(values[2:6]).reshape(2, 2)),
            Factor((1,), jnp.asarray(values[6:8]) + jnp.asarray(values[8:10])),
        ),
    )
    bp = exact_tree_sum_product(graph)
    exact = enumerate_distribution(graph)
    for actual, expected in zip(bp.variable_beliefs, exact.variable_marginals, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-11, atol=1e-11)


def test_enumeration_handles_noncanonical_factor_axis_order() -> None:
    graph = DiscreteFactorGraph(
        cardinalities=(2, 3),
        factors=(Factor((1, 0), jnp.arange(1.0, 7.0).reshape(3, 2)),),
    )
    exact = enumerate_distribution(graph)
    expected = jnp.arange(1.0, 7.0).reshape(3, 2).T
    expected = expected / jnp.sum(expected)
    np.testing.assert_allclose(np.asarray(exact.joint), np.asarray(expected), atol=1e-12)

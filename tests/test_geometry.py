# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kl_projection_message_passing.geometry import (
    consensus_product_operator,
    coordinate_marginal,
    diagonal_i_projection,
    kl_divergence,
    normalize,
    product_reverse_kl_projection,
    pythagorean_gap,
)


@given(st.lists(st.floats(min_value=0.01, max_value=10), min_size=2, max_size=20))
@settings(max_examples=20, deadline=None)
def test_normalize_returns_a_simplex(values: list[float]) -> None:
    probability = normalize(jnp.asarray(values))
    np.testing.assert_allclose(np.asarray(probability).sum(), 1.0, atol=1e-12)
    assert np.all(np.asarray(probability) > 0)


def test_diagonal_projection_is_normalized_restriction() -> None:
    joint = normalize(jnp.arange(1.0, 28.0).reshape(3, 3, 3))
    projected = diagonal_i_projection(joint)
    expected_diagonal = normalize(jnp.asarray([joint[0, 0, 0], joint[1, 1, 1], joint[2, 2, 2]]))

    np.testing.assert_allclose(np.asarray(projected).sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(projected[(jnp.arange(3),) * 3]), np.asarray(expected_diagonal), atol=1e-12
    )
    assert np.count_nonzero(np.asarray(projected)) == 3


def test_diagonal_projection_satisfies_primal_pythagorean_identity() -> None:
    original = normalize(jnp.asarray([[1.0, 2.0], [3.0, 7.0]]))
    projection = diagonal_i_projection(original)
    reference = jnp.asarray([[0.2, 0.0], [0.0, 0.8]])
    np.testing.assert_allclose(
        np.asarray(pythagorean_gap(reference, original, projection)), 0.0, atol=1e-12
    )


def test_product_projection_preserves_all_coordinate_marginals() -> None:
    joint = normalize(jnp.arange(1.0, 25.0).reshape(2, 3, 4))
    projected = product_reverse_kl_projection(joint)
    np.testing.assert_allclose(np.asarray(projected).sum(), 1.0, atol=1e-12)
    for coordinate in range(joint.ndim):
        np.testing.assert_allclose(
            np.asarray(coordinate_marginal(projected, coordinate)),
            np.asarray(coordinate_marginal(joint, coordinate)),
            atol=1e-12,
        )


def test_product_projection_minimizes_forward_kl_over_product_candidates() -> None:
    joint = normalize(jnp.asarray([[0.35, 0.15], [0.1, 0.4]]))
    optimum = product_reverse_kl_projection(joint)
    candidate = jnp.outer(jnp.asarray([0.8, 0.2]), jnp.asarray([0.25, 0.75]))
    assert float(kl_divergence(joint, optimum)) <= float(kl_divergence(joint, candidate))


def test_consensus_product_operator_is_jittable_and_finite() -> None:
    operator = jax.jit(consensus_product_operator)
    result = operator(jnp.asarray([[0.35, 0.15], [0.1, 0.4]]))
    assert bool(jnp.all(jnp.isfinite(result.factorized)))
    assert float(result.consensus_residual) >= 0
    assert float(result.factorization_residual) >= 0


def test_kl_divergence_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        kl_divergence(jnp.ones((2,)), jnp.ones((2, 1)))


def test_diagonal_projection_rejects_unequal_alphabets() -> None:
    with pytest.raises(ValueError, match="equal size"):
        diagonal_i_projection(jnp.ones((2, 3)))

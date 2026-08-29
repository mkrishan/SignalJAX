# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kl_projection_message_passing.projection import (
    ProjectionStage,
    affine_projector,
    box_projector,
    consensus_projector,
    cyclic_projections,
    euclidean_residual_corollary,
    projection_residual,
    relaxed_projection,
)


def test_affine_projector_is_idempotent_and_jittable() -> None:
    projector = affine_projector(jnp.asarray([[1.0, 1.0]]), jnp.asarray([1.0]))
    point = jnp.asarray([2.0, -0.5])
    projected = jax.jit(projector)(point)
    np.testing.assert_allclose(np.asarray(jnp.sum(projected)), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(projector(projected)), np.asarray(projected), atol=1e-12)


def test_euclidean_residual_step_is_cyclic_projection_corollary() -> None:
    projector = affine_projector(jnp.asarray([[1.0, 1.0]]), jnp.asarray([1.0]))
    result = euclidean_residual_corollary(jnp.asarray([2.0, -0.5]), projector)
    np.testing.assert_allclose(np.asarray(result.gradient_identity_error), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(result.cyclic_projection_error), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result.unit_sgd_state), np.asarray(result.projected_state)
    )


def test_cyclic_projections_reach_two_affine_constraints() -> None:
    x_zero = affine_projector(jnp.asarray([[1.0, 0.0]]), jnp.asarray([0.0]))
    y_one = affine_projector(jnp.asarray([[0.0, 1.0]]), jnp.asarray([1.0]))
    trajectory = cyclic_projections(
        (ProjectionStage("x=0", x_zero), ProjectionStage("y=1", y_one)),
        jnp.asarray([3.0, -2.0]),
        sweeps=2,
    )
    np.testing.assert_allclose(np.asarray(trajectory.states[-1]), [0.0, 1.0], atol=1e-12)
    assert trajectory.stage_names[-1] == "sweep-2:y=1"
    assert trajectory.residual_norms.shape == (4,)


def test_relaxation_box_and_consensus_projections() -> None:
    box = box_projector(jnp.asarray([-1.0, 0.0]), jnp.asarray([1.0, 2.0]))
    point = jnp.asarray([3.0, -1.0])
    np.testing.assert_allclose(np.asarray(box(point)), [1.0, 0.0])
    np.testing.assert_allclose(np.asarray(projection_residual(point, box)), [-2.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(relaxed_projection(point, box, relaxation=0.5)), [2.0, -0.5]
    )

    consensus = consensus_projector(
        (jnp.asarray([0.0, 2.0]), jnp.asarray([2.0, 0.0])), weights=(1.0, 3.0)
    )
    np.testing.assert_allclose(np.asarray(consensus[0]), [1.5, 0.5])
    np.testing.assert_allclose(np.asarray(consensus[1]), [1.5, 0.5])


def test_projection_validation() -> None:
    with pytest.raises(ValueError, match="rank two"):
        affine_projector(jnp.ones((2,)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="right-hand side"):
        affine_projector(jnp.ones((2, 2)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="lower bounds"):
        box_projector(jnp.asarray([1.0]), jnp.asarray([0.0]))
    with pytest.raises(ValueError, match="at least one"):
        consensus_projector(())
    with pytest.raises(ValueError, match="same shape"):
        consensus_projector((jnp.ones((1,)), jnp.ones((2,))))
    with pytest.raises(ValueError, match="at least one stage"):
        cyclic_projections((), jnp.asarray(1.0), sweeps=1)
    with pytest.raises(ValueError, match="positive"):
        cyclic_projections(
            (ProjectionStage("id", lambda value: value),), jnp.asarray(1.0), sweeps=0
        )

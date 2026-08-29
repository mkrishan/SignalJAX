# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from kl_projection_message_passing.operators import (
    ComposedOperator,
    OperatorStage,
    compare_finite_and_differential,
    fixed_point_residual,
    iterate_operator,
    relaxed_operator_step,
    tree_add,
    tree_l2_norm,
    tree_scale,
    tree_subtract,
)


def test_pytree_arithmetic_and_norm() -> None:
    left = {"a": jnp.asarray([1.0, 2.0]), "b": jnp.asarray(3.0)}
    right = {"a": jnp.asarray([0.5, 1.0]), "b": jnp.asarray(-1.0)}
    added = tree_add(left, right)
    subtracted = tree_subtract(left, right)
    scaled = tree_scale(2.0, right)
    np.testing.assert_allclose(np.asarray(added["a"]), [1.5, 3.0])
    np.testing.assert_allclose(np.asarray(subtracted["b"]), 4.0)
    np.testing.assert_allclose(np.asarray(scaled["a"]), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(tree_l2_norm(left)), np.sqrt(14.0))
    with pytest.raises(ValueError, match="at least one leaf"):
        tree_l2_norm({})


def test_composed_operator_trace_and_differential() -> None:
    operator = ComposedOperator(
        (
            OperatorStage("scale", lambda value: 2.0 * value, "local-relation"),
            OperatorStage("shift", lambda value: value + 1.0, "consensus"),
        ),
        name="toy",
    )
    trace = operator.trace(jnp.asarray([1.0, 3.0]))
    assert trace.stage_names == ("input", "scale", "shift")
    np.testing.assert_allclose(np.asarray(trace.final_state), [3.0, 7.0])
    differential = operator.differential(jnp.asarray([1.0, 3.0]), jnp.asarray([0.5, -1.0]))
    np.testing.assert_allclose(np.asarray(differential), [1.0, -2.0])


def test_operator_iteration_and_relaxation() -> None:
    def operator(value: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * value

    trace = iterate_operator(operator, jnp.asarray(8.0), steps=3)
    np.testing.assert_allclose(np.asarray(jnp.stack(trace.states)), [8.0, 4.0, 2.0, 1.0])
    np.testing.assert_allclose(np.asarray(trace.fixed_point_residuals), [4.0, 2.0, 1.0])
    np.testing.assert_allclose(np.asarray(trace.step_norms), [4.0, 2.0, 1.0])
    np.testing.assert_allclose(np.asarray(fixed_point_residual(operator, jnp.asarray(8.0))), 4.0)
    np.testing.assert_allclose(
        np.asarray(relaxed_operator_step(operator, jnp.asarray(8.0), 0.5)), 6.0
    )
    with pytest.raises(ValueError, match="positive"):
        iterate_operator(operator, jnp.asarray(1.0), steps=0)
    with pytest.raises(ValueError, match="relaxation"):
        iterate_operator(operator, jnp.asarray(1.0), steps=1, relaxation=2.5)


def test_central_finite_difference_converges_to_operator_jvp() -> None:
    comparison = compare_finite_and_differential(
        lambda value: value**3,
        jnp.asarray(1.2),
        jnp.asarray(0.7),
        epsilons=(1e-2, 5e-3, 2.5e-3),
    )
    np.testing.assert_allclose(np.asarray(comparison.differential), 3 * 1.2**2 * 0.7)
    assert float(comparison.absolute_errors[-1]) < float(comparison.absolute_errors[0]) / 10
    with pytest.raises(ValueError, match="non-empty"):
        compare_finite_and_differential(
            lambda value: value, jnp.asarray(1.0), jnp.asarray(1.0), epsilons=()
        )
    with pytest.raises(ValueError, match="positive"):
        compare_finite_and_differential(
            lambda value: value, jnp.asarray(1.0), jnp.asarray(1.0), epsilons=(-1.0,)
        )


def test_operator_validation() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        ComposedOperator(())
    stage = OperatorStage("same", lambda value: value)
    with pytest.raises(ValueError, match="unique"):
        ComposedOperator((stage, stage))
    with pytest.raises(TypeError, match="not callable"):
        OperatorStage("bad", None)  # type: ignore[arg-type]

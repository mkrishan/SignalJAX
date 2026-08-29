# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kl_projection_message_passing.deterministic import (
    DeterministicGraph,
    Operation,
    central_log_message_difference,
    deterministic_factor_message,
    log_message_differential,
    log_message_directional_derivative,
)


def branching_graph() -> DeterministicGraph:
    return DeterministicGraph(
        inputs=("x",),
        operations=(
            Operation("y1", ("x",), jnp.sin),
            Operation("y2", ("x",), lambda x: x**2),
            Operation("z", ("y1", "y2"), lambda y1, y2: y1 + 3.0 * y2),
        ),
        output="z",
    )


def test_path_additivity_matches_monolithic_gradient() -> None:
    graph = branching_graph()
    x = jnp.asarray(0.4)
    readout = graph.readout({"x": x}, log_potential=lambda z: -z)
    expected = jax.grad(lambda value: -(jnp.sin(value) + 3.0 * value**2))(x)
    np.testing.assert_allclose(np.asarray(readout.adjoints["x"]), np.asarray(expected), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(readout.adjoints["x"]), np.asarray(-(jnp.cos(x) + 6.0 * x)), atol=1e-12
    )


def test_vector_cotangent_seed_matches_jax_vjp() -> None:
    graph = DeterministicGraph(
        inputs=("x",),
        operations=(Operation("z", ("x",), lambda x: jnp.stack((x**2, jnp.sin(x)))),),
        output="z",
    )
    x = jnp.asarray(0.7)
    cotangent = jnp.asarray([2.0, -0.3])
    readout = graph.readout({"x": x}, output_cotangent=cotangent)
    _, pullback = jax.vjp(lambda value: jnp.stack((value**2, jnp.sin(value))), x)
    expected = pullback(cotangent)[0]
    np.testing.assert_allclose(np.asarray(readout.adjoints["x"]), np.asarray(expected), atol=1e-12)


def test_local_log_message_identity_is_the_chain_rule() -> None:
    parents = (jnp.asarray(0.3), jnp.asarray(-0.8))

    def local_map(x1: jax.Array, x2: jax.Array) -> jax.Array:
        return x1 * x2 + jnp.sin(x1)

    def downstream(y: jax.Array) -> jax.Array:
        return jnp.exp(-0.5 * (y - 1.2) ** 2)

    message = deterministic_factor_message(local_map, parents, 0, downstream)

    actual = log_message_differential(message, parents[0])
    y = local_map(*parents)
    downstream_score = jax.grad(lambda value: jnp.log(downstream(value)))(y)
    local_jacobian = jax.grad(lambda x1: local_map(x1, parents[1]))(parents[0])
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(downstream_score * local_jacobian), atol=1e-12
    )


def test_finite_difference_converges_to_directional_readout() -> None:
    def message(x: jax.Array) -> jax.Array:
        return jnp.exp(jnp.sin(x) - 0.2 * x**2)

    point = jnp.asarray(0.6)
    direction = jnp.asarray(-1.7)
    exact = log_message_directional_derivative(message, point, direction)
    coarse = central_log_message_difference(message, point, direction, 1e-2)
    fine = central_log_message_difference(message, point, direction, 1e-4)
    assert abs(float(fine - exact)) < abs(float(coarse - exact))
    np.testing.assert_allclose(np.asarray(fine), np.asarray(exact), atol=1e-7)


def test_explicit_readout_can_be_jitted_as_a_static_graph() -> None:
    graph = branching_graph()
    compiled = jax.jit(lambda x: graph.readout({"x": x}).adjoints["x"])
    x = jnp.asarray(0.25)
    np.testing.assert_allclose(
        np.asarray(compiled(x)), np.asarray(jnp.cos(x) + 6.0 * x), atol=1e-12
    )

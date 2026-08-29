# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Section 4.1: two forward paths become an additive reverse readout."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kl_projection_message_passing import DeterministicGraph, Operation


def main() -> None:
    graph = DeterministicGraph(
        inputs=("x",),
        operations=(
            Operation("y1", ("x",), jnp.sin),
            Operation("y2", ("x",), lambda x: x**2),
            Operation("z", ("y1", "y2"), lambda y1, y2: y1 + 3.0 * y2),
        ),
        output="z",
    )
    x = jnp.asarray(0.4)
    readout = graph.readout({"x": x}, log_potential=lambda z: -z)
    direct = jax.grad(lambda value: -(jnp.sin(value) + 3.0 * value**2))(x)

    print(f"forward output: {float(readout.values['z']):.8f}")
    print(f"message readout: {float(readout.adjoints['x']):.8f}")
    print(f"direct gradient: {float(direct):.8f}")
    print(f"absolute error: {abs(float(readout.adjoints['x'] - direct)):.3e}")


if __name__ == "__main__":
    main()

# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Run compact numerical checks for the companion package's main identities."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from kl_projection_message_passing import (
    DeterministicGraph,
    DiscreteFactorGraph,
    Factor,
    Indicator,
    Operation,
    Product,
    Sum,
    SumProductNetwork,
    affine_projector,
    analyze_signal_sequence,
    compare_finite_and_differential,
    consensus_product_operator,
    enumerate_distribution,
    enumerate_posterior,
    euclidean_residual_corollary,
    exact_tree_sum_product,
    merge_unfolded_downward,
    unfold_spn,
)


def deterministic_error() -> jax.Array:
    graph = DeterministicGraph(
        inputs=("x",),
        operations=(
            Operation("left", ("x",), jnp.sin),
            Operation("right", ("x",), lambda x: x**2),
            Operation("output", ("left", "right"), lambda left, right: left + right),
        ),
        output="output",
    )
    x = jnp.asarray(0.4)
    readout = graph.readout({"x": x}).adjoints["x"]
    direct = jax.grad(lambda value: jnp.sin(value) + value**2)(x)
    return jnp.abs(readout - direct)


def factor_graph_error() -> jax.Array:
    graph = DiscreteFactorGraph(
        cardinalities=(2, 2),
        factors=(
            Factor((0,), jnp.asarray([0.6, 0.4])),
            Factor((0, 1), jnp.asarray([[2.0, 0.5], [0.3, 1.7]])),
            Factor((1,), jnp.asarray([0.2, 0.8])),
        ),
    )
    bp = exact_tree_sum_product(graph)
    exact = enumerate_distribution(graph)
    return jnp.max(
        jnp.stack(
            tuple(
                jnp.max(jnp.abs(actual - expected))
                for actual, expected in zip(
                    bp.variable_beliefs, exact.variable_marginals, strict=True
                )
            )
        )
    )


def spn_errors() -> tuple[jax.Array, jax.Array]:
    spn = SumProductNetwork(
        cardinalities=(2, 2),
        nodes=(
            Indicator(0, 0),
            Indicator(0, 1),
            Sum((0, 1), jnp.asarray([0.6, 0.4])),
            Indicator(1, 0),
            Indicator(1, 1),
            Product((2, 3)),
            Product((2, 4)),
            Sum((5, 6), jnp.asarray([0.7, 0.3])),
        ),
        root=7,
    )
    evidence = (jnp.asarray([0.4, 0.9]), jnp.asarray([0.7, 0.2]))
    readout = spn.upward_downward(evidence)
    exact = enumerate_posterior(spn, evidence)
    marginal_error = jnp.max(
        jnp.stack(
            tuple(
                jnp.max(jnp.abs(actual - expected))
                for actual, expected in zip(
                    readout.variable_marginals, exact.variable_marginals, strict=True
                )
            )
        )
    )
    unfolded, origin = unfold_spn(spn)
    merged = merge_unfolded_downward(spn, unfolded.upward_downward(evidence), origin)
    merge_error = jnp.max(
        jnp.stack(
            tuple(
                jnp.abs(actual - expected)
                for actual, expected in zip(merged, readout.downward, strict=True)
            )
        )
    )
    return marginal_error, merge_error


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    projection = consensus_product_operator(jnp.asarray([[0.35, 0.15], [0.1, 0.4]]))
    spn_marginal_error, unfolding_merge_error = spn_errors()
    differential = compare_finite_and_differential(
        lambda value: value**3,
        jnp.asarray(1.2),
        jnp.asarray(0.7),
        epsilons=(1e-3,),
    )
    projector = affine_projector(jnp.asarray([[1.0, 1.0]]), jnp.asarray([1.0]))
    corollary = euclidean_residual_corollary(jnp.asarray([2.0, -0.5]), projector)
    attenuation = analyze_signal_sequence(
        (jnp.asarray(1.0), jnp.asarray(0.5), jnp.asarray(0.25)),
        name="certified-toy-signal",
        analytic_local_bounds=(0.5, 0.5),
    )
    report = {
        "consensus_residual": float(projection.consensus_residual),
        "cyclic_projection_corollary_error": float(corollary.cyclic_projection_error),
        "deterministic_readout_error": float(deterministic_error()),
        "finite_differential_error": float(differential.absolute_errors[0]),
        "factor_graph_marginal_error": float(factor_graph_error()),
        "factorization_residual": float(projection.factorization_residual),
        "spn_marginal_error": float(spn_marginal_error),
        "toy_signal_certified_exponential": attenuation.certified_exponential,
        "unfolding_merge_error": float(unfolding_merge_error),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

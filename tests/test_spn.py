# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kl_projection_message_passing.spn import (
    Indicator,
    Product,
    Sum,
    SumProductNetwork,
    enumerate_posterior,
    log_evidence_gradient,
    merge_unfolded_downward,
    unfold_spn,
)


def diagonal_mixture_spn() -> SumProductNetwork:
    return SumProductNetwork(
        cardinalities=(2, 2),
        nodes=(
            Indicator(0, 0),
            Indicator(0, 1),
            Indicator(1, 0),
            Indicator(1, 1),
            Product((0, 2)),
            Product((1, 3)),
            Sum((4, 5), jnp.asarray([0.7, 0.3])),
        ),
        root=6,
    )


def shared_subcircuit_spn() -> SumProductNetwork:
    return SumProductNetwork(
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


def test_spn_readouts_match_enumerated_posterior() -> None:
    spn = diagonal_mixture_spn()
    evidence = (jnp.asarray([0.8, 0.2]), jnp.asarray([0.3, 0.9]))
    readout = spn.upward_downward(evidence)
    enumeration = enumerate_posterior(spn, evidence)

    for actual, expected in zip(
        readout.variable_marginals, enumeration.variable_marginals, strict=True
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)
        np.testing.assert_allclose(np.asarray(actual).sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(readout.root_value), np.asarray(enumeration.partition), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(readout.gate_conditionals[spn.root]).sum(), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(readout.gate_joint_masses[spn.root]),
        np.asarray(readout.gate_conditionals[spn.root]),
        atol=1e-12,
    )


def test_log_evidence_differential_equals_variable_marginals() -> None:
    spn = diagonal_mixture_spn()
    evidence = (jnp.asarray([0.8, 0.2]), jnp.asarray([0.3, 0.9]))
    gradient = log_evidence_gradient(spn, tuple(jnp.log(item) for item in evidence))
    readout = spn.upward_downward(evidence)
    for actual, expected in zip(gradient, readout.variable_marginals, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


def test_unfolded_contexts_merge_to_shared_dag_contexts() -> None:
    spn = shared_subcircuit_spn()
    evidence = (jnp.asarray([0.4, 0.9]), jnp.asarray([0.7, 0.2]))
    dag_readout = spn.upward_downward(evidence)
    unfolded, origin = unfold_spn(spn)
    unfolded_readout = unfolded.upward_downward(evidence)
    merged = merge_unfolded_downward(spn, unfolded_readout, origin)

    np.testing.assert_allclose(
        np.asarray(unfolded_readout.root_value), np.asarray(dag_readout.root_value), atol=1e-12
    )
    for actual, expected in zip(merged, dag_readout.downward, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)
    enumeration = enumerate_posterior(spn, evidence)
    for actual, expected in zip(
        dag_readout.variable_marginals, enumeration.variable_marginals, strict=True
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


def test_spn_readout_is_jittable_with_static_structure() -> None:
    spn = diagonal_mixture_spn()
    compiled = jax.jit(lambda evidence: spn.upward_downward(evidence).variable_marginals)
    evidence = (jnp.asarray([0.8, 0.2]), jnp.asarray([0.3, 0.9]))
    actual = compiled(evidence)
    expected = spn.upward_downward(evidence).variable_marginals
    for compiled_marginal, direct_marginal in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(compiled_marginal), np.asarray(direct_marginal), atol=1e-12
        )


def test_incomplete_and_non_decomposable_circuits_are_rejected() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        SumProductNetwork(
            cardinalities=(2, 2),
            nodes=(Indicator(0, 0), Indicator(1, 0), Sum((0, 1), jnp.ones((2,)))),
            root=2,
        )


def test_evidence_shapes_are_validated() -> None:
    spn = diagonal_mixture_spn()
    with pytest.raises(ValueError, match="expected"):
        spn.upward_downward((jnp.ones((3,)), jnp.ones((2,))))
    with pytest.raises(ValueError, match="not decomposable"):
        SumProductNetwork(
            cardinalities=(2,),
            nodes=(Indicator(0, 0), Indicator(0, 1), Product((0, 1))),
            root=2,
        )

# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from kl_projection_message_passing.signals import (
    SignalChannel,
    SignalGranularity,
    analyze_signal_sequence,
    perturbation_signal_trace,
    propagate_jvp_signal,
    propagate_vjp_signal,
    readout_signal_trace,
)


def test_observation_is_not_mislabeled_as_exponential_certificate() -> None:
    observed = analyze_signal_sequence(
        (jnp.asarray(8.0), jnp.asarray(4.0), jnp.asarray(2.0)),
        name="targets",
        channel=SignalChannel.PROJECTION_TARGET,
        granularity=SignalGranularity.LAYER,
    )
    assert observed.observed_strict_attenuation
    assert not observed.certified_exponential
    np.testing.assert_allclose(np.asarray(observed.local_gains), [0.5, 0.5])

    certified = analyze_signal_sequence(
        observed.signals,
        name="targets",
        channel=SignalChannel.PROJECTION_TARGET,
        analytic_local_bounds=(0.6, 0.6),
    )
    assert certified.certified_exponential
    np.testing.assert_allclose(np.asarray(certified.certified_uniform_bound), 0.6)


def test_amplification_and_zero_signal_cases_are_explicit() -> None:
    amplification = analyze_signal_sequence(
        (jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(1.0)), name="mixed"
    )
    assert amplification.observed_amplification
    assert not amplification.observed_monotone_nonincrease

    zero = analyze_signal_sequence(
        (jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0)), name="zero"
    )
    np.testing.assert_allclose(np.asarray(zero.local_gains[0]), 1.0)
    assert np.isinf(float(zero.local_gains[1]))


def test_state_perturbation_and_readout_traces() -> None:
    left = (jnp.asarray([2.0, 0.0]), jnp.asarray([1.0, 0.0]))
    right = (jnp.zeros((2,)), jnp.zeros((2,)))
    perturbation = perturbation_signal_trace(left, right, name="separation")
    np.testing.assert_allclose(np.asarray(perturbation.norms), [2.0, 1.0])

    readout = readout_signal_trace(
        left,
        lambda state: state[0],
        name="first-coordinate",
        channel=SignalChannel.LOCAL_SCORE,
    )
    np.testing.assert_allclose(np.asarray(readout.norms), [2.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        perturbation_signal_trace(left, right[:1], name="bad")


def test_jvp_and_vjp_propagate_through_local_stages() -> None:
    stages = (
        ("scale-two", lambda value: 2.0 * value),
        ("scale-three", lambda value: 3.0 * value),
    )
    forward = propagate_jvp_signal(stages, jnp.asarray(1.0), jnp.asarray(0.5))
    assert forward.labels == ("input", "scale-two", "scale-three")
    np.testing.assert_allclose(np.asarray(jnp.stack(forward.signals)), [0.5, 1.0, 3.0])

    reverse = propagate_vjp_signal(stages, jnp.asarray(1.0), jnp.asarray(1.0))
    assert reverse.labels == ("scale-three", "scale-two", "input")
    np.testing.assert_allclose(np.asarray(jnp.stack(reverse.signals)), [1.0, 3.0, 6.0])


def test_signal_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        analyze_signal_sequence((), name="empty")
    with pytest.raises(ValueError, match="same length"):
        analyze_signal_sequence((jnp.asarray(1.0),), name="bad", labels=("a", "b"))
    with pytest.raises(ValueError, match="expected"):
        analyze_signal_sequence(
            (jnp.asarray(1.0), jnp.asarray(0.5)),
            name="bad-bounds",
            analytic_local_bounds=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        analyze_signal_sequence(
            (jnp.asarray(1.0), jnp.asarray(0.5)),
            name="bad-bounds",
            analytic_local_bounds=(-0.5,),
        )

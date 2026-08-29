# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Signal transfer, amplification, attenuation, and contraction diagnostics.

A signal is any JAX pytree selected by a paradigm-specific readout.  It may be a cotangent, a finite
target displacement, a projection residual, a marginal perturbation, an energy error, or a local
forward score.  The analysis deliberately records the channel and granularity instead of conflating
these objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .operators import PyTree, tree_l2_norm, tree_subtract

Array = jax.Array
Transform = Callable[[PyTree], PyTree]
Readout = Callable[[PyTree], PyTree]


class SignalChannel(str, Enum):
    """Semantic channel from which a propagated signal was read."""

    COTANGENT = "cotangent"
    PROJECTION_TARGET = "projection-target"
    PROJECTION_RESIDUAL = "projection-residual"
    MARGINAL = "marginal"
    ENERGY_ERROR = "energy-error"
    LAYERWISE_TARGET = "layerwise-target"
    LOCAL_SCORE = "local-score"
    STATE_PERTURBATION = "state-perturbation"
    USER_DEFINED = "user-defined"


class SignalGranularity(str, Enum):
    """Representational scale at which a signal was sampled."""

    NODE = "node"
    EDGE = "edge"
    LAYER = "layer"
    BLOCK = "block"
    ITERATION = "iteration"
    GRAPH = "graph"


@dataclass(frozen=True)
class SignalTrace:
    """Norm and gain diagnostics for an ordered signal sequence.

    ``certified_exponential`` is true only when independent analytic local bounds were supplied and
    their uniform maximum is strictly below one.  A decreasing empirical trace alone is never
    promoted to a theorem.
    """

    name: str
    channel: SignalChannel
    granularity: SignalGranularity
    labels: tuple[str, ...]
    signals: tuple[PyTree, ...]
    norms: Array
    local_gains: Array
    cumulative_gains: Array
    log_local_gains: Array
    observed_monotone_nonincrease: bool
    observed_strict_attenuation: bool
    observed_amplification: bool
    empirical_uniform_gain: Array
    empirical_log_slope: Array
    empirical_log_fit_r_squared: Array
    analytic_local_bounds: Array | None
    certified_uniform_bound: Array
    certified_exponential: bool


def _safe_ratios(numerators: Array, denominators: Array, atol: float) -> Array:
    both_zero = (numerators <= atol) & (denominators <= atol)
    positive_denominator = denominators > atol
    ratio = jnp.where(positive_denominator, numerators / denominators, jnp.inf)
    return jnp.where(both_zero, 1.0, ratio)


def _log_linear_fit(norms: Array, atol: float) -> tuple[Array, Array]:
    if norms.size < 2:
        nan = jnp.asarray(jnp.nan, dtype=norms.dtype)
        return nan, nan
    depth = jnp.arange(norms.size, dtype=norms.dtype)
    log_norms = jnp.log(jnp.maximum(norms, jnp.asarray(atol, dtype=norms.dtype)))
    centered_depth = depth - jnp.mean(depth)
    centered_logs = log_norms - jnp.mean(log_norms)
    denominator = jnp.sum(centered_depth**2)
    slope = jnp.sum(centered_depth * centered_logs) / denominator
    prediction = jnp.mean(log_norms) + slope * centered_depth
    residual_sum = jnp.sum((log_norms - prediction) ** 2)
    total_sum = jnp.sum(centered_logs**2)
    r_squared = jnp.where(total_sum > 0, 1.0 - residual_sum / total_sum, 1.0)
    return slope, r_squared


def analyze_signal_sequence(
    signals: Sequence[PyTree],
    *,
    name: str,
    channel: SignalChannel = SignalChannel.USER_DEFINED,
    granularity: SignalGranularity = SignalGranularity.NODE,
    labels: Sequence[str] | None = None,
    analytic_local_bounds: Sequence[float] | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-9,
) -> SignalTrace:
    """Analyze an arbitrary ordered sequence of paradigm-specific signals."""

    signal_tuple = tuple(signals)
    if not signal_tuple:
        raise ValueError("a signal trace needs at least one signal")
    if not name:
        raise ValueError("signal trace names must be non-empty")
    label_tuple = (
        tuple(labels)
        if labels is not None
        else tuple(str(index) for index in range(len(signal_tuple)))
    )
    if len(label_tuple) != len(signal_tuple):
        raise ValueError("labels and signals must have the same length")

    norms = jnp.stack(tuple(tree_l2_norm(signal) for signal in signal_tuple))
    local_gains = _safe_ratios(norms[1:], norms[:-1], atol)
    cumulative_gains = _safe_ratios(norms, jnp.full_like(norms, norms[0]), atol)
    log_local_gains = jnp.log(local_gains)
    tolerance = atol + rtol * norms[:-1]
    observed_monotone = bool(jnp.all(norms[1:] <= norms[:-1] + tolerance))
    observed_strict = bool(jnp.all(norms[1:] < norms[:-1] - tolerance))
    observed_amplification = bool(jnp.any(norms[1:] > norms[:-1] + tolerance))
    empirical_uniform_gain = (
        jnp.max(local_gains) if local_gains.size else jnp.asarray(jnp.nan, dtype=norms.dtype)
    )
    log_slope, log_r_squared = _log_linear_fit(norms, atol)

    if analytic_local_bounds is None:
        bounds = None
        certified_bound = jnp.asarray(jnp.nan, dtype=norms.dtype)
        certified_exponential = False
    else:
        bounds = jnp.asarray(tuple(analytic_local_bounds), dtype=norms.dtype)
        if bounds.shape != local_gains.shape:
            raise ValueError(
                f"analytic bounds have shape {bounds.shape}; expected {local_gains.shape}"
            )
        if bool(jnp.any(bounds < 0)):
            raise ValueError("analytic local bounds must be nonnegative")
        certified_bound = (
            jnp.max(bounds) if bounds.size else jnp.asarray(jnp.nan, dtype=norms.dtype)
        )
        certified_exponential = bool(bounds.size and certified_bound < 1.0)

    return SignalTrace(
        name=name,
        channel=channel,
        granularity=granularity,
        labels=label_tuple,
        signals=signal_tuple,
        norms=norms,
        local_gains=local_gains,
        cumulative_gains=cumulative_gains,
        log_local_gains=log_local_gains,
        observed_monotone_nonincrease=observed_monotone,
        observed_strict_attenuation=observed_strict,
        observed_amplification=observed_amplification,
        empirical_uniform_gain=empirical_uniform_gain,
        empirical_log_slope=log_slope,
        empirical_log_fit_r_squared=log_r_squared,
        analytic_local_bounds=bounds,
        certified_uniform_bound=certified_bound,
        certified_exponential=certified_exponential,
    )


def perturbation_signal_trace(
    left_states: Sequence[PyTree],
    right_states: Sequence[PyTree],
    *,
    name: str,
    channel: SignalChannel = SignalChannel.STATE_PERTURBATION,
    granularity: SignalGranularity = SignalGranularity.ITERATION,
    labels: Sequence[str] | None = None,
    analytic_local_bounds: Sequence[float] | None = None,
) -> SignalTrace:
    """Analyze separation between two state trajectories."""

    left = tuple(left_states)
    right = tuple(right_states)
    if len(left) != len(right):
        raise ValueError("state trajectories must have the same length")
    differences = tuple(tree_subtract(a, b) for a, b in zip(left, right, strict=True))
    return analyze_signal_sequence(
        differences,
        name=name,
        channel=channel,
        granularity=granularity,
        labels=labels,
        analytic_local_bounds=analytic_local_bounds,
    )


def readout_signal_trace(
    states: Sequence[PyTree],
    readout: Readout,
    *,
    name: str,
    channel: SignalChannel,
    granularity: SignalGranularity = SignalGranularity.ITERATION,
    labels: Sequence[str] | None = None,
    analytic_local_bounds: Sequence[float] | None = None,
) -> SignalTrace:
    """Apply a paradigm-specific readout to a state trajectory and analyze its signals."""

    return analyze_signal_sequence(
        tuple(readout(state) for state in states),
        name=name,
        channel=channel,
        granularity=granularity,
        labels=labels,
        analytic_local_bounds=analytic_local_bounds,
    )


class LocalPropagation(NamedTuple):
    """Primal states and propagated JVP signals through named local maps."""

    states: tuple[PyTree, ...]
    signals: tuple[PyTree, ...]
    labels: tuple[str, ...]


def propagate_jvp_signal(
    stages: Sequence[tuple[str, Transform]], initial_state: PyTree, initial_signal: PyTree
) -> LocalPropagation:
    """Propagate one tangent through any ordered family of local maps."""

    states = [initial_state]
    signals = [initial_signal]
    labels = ["input"]
    state = initial_state
    signal = initial_signal
    for name, transform in stages:
        if not name or not callable(transform):
            raise ValueError("every local propagation stage needs a name and callable map")
        state, signal = jax.jvp(transform, (state,), (signal,))
        states.append(state)
        signals.append(signal)
        labels.append(name)
    return LocalPropagation(tuple(states), tuple(signals), tuple(labels))


def propagate_vjp_signal(
    stages: Sequence[tuple[str, Transform]], initial_state: PyTree, output_signal: PyTree
) -> LocalPropagation:
    """Propagate one cotangent through local maps, retaining output-to-input order."""

    states = [initial_state]
    pullbacks: list[Callable[[PyTree], tuple[PyTree]]] = []
    labels = ["input"]
    state = initial_state
    for name, transform in stages:
        if not name or not callable(transform):
            raise ValueError("every local propagation stage needs a name and callable map")
        state, pullback = jax.vjp(transform, state)
        states.append(state)
        pullbacks.append(pullback)
        labels.append(name)

    reverse_signals = [output_signal]
    signal = output_signal
    for pullback in reversed(pullbacks):
        signal = pullback(signal)[0]
        reverse_signals.append(signal)
    return LocalPropagation(
        states=tuple(reversed(states)),
        signals=tuple(reverse_signals),
        labels=tuple(reversed(labels)),
    )

# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Compare certified cotangent contraction with an empirical target trace."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from kl_projection_message_passing import (
    SignalChannel,
    SignalGranularity,
    analyze_signal_sequence,
    propagate_vjp_signal,
)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    stages = (
        ("layer-1", lambda value: 0.8 * value),
        ("layer-2", lambda value: 0.6 * value),
        ("layer-3", lambda value: 0.5 * value),
    )
    cotangents = propagate_vjp_signal(stages, jnp.asarray(1.0), jnp.asarray(1.0))
    gradient_trace = analyze_signal_sequence(
        cotangents.signals,
        name="reverse cotangents",
        channel=SignalChannel.COTANGENT,
        granularity=SignalGranularity.LAYER,
        labels=cotangents.labels,
        analytic_local_bounds=(0.5, 0.6, 0.8),
    )

    target_trace = analyze_signal_sequence(
        (jnp.asarray(1.0), jnp.asarray(0.7), jnp.asarray(0.55), jnp.asarray(0.6)),
        name="finite layerwise targets",
        channel=SignalChannel.PROJECTION_TARGET,
        granularity=SignalGranularity.LAYER,
        labels=("output", "layer-3", "layer-2", "layer-1"),
    )
    report = {
        "cotangent_certified_exponential": gradient_trace.certified_exponential,
        "cotangent_norms": [float(value) for value in gradient_trace.norms],
        "cotangent_uniform_bound": float(gradient_trace.certified_uniform_bound),
        "target_has_amplification": target_trace.observed_amplification,
        "target_norms": [float(value) for value in target_trace.norms],
        "target_certified_exponential": target_trace.certified_exponential,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

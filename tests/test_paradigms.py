# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from kl_projection_message_passing.paradigms import (
    ApproximationFamily,
    ClaimLevel,
    ExecutableParadigm,
    Geometry,
    LocalRelation,
    ParadigmRegistry,
    ParadigmSpec,
    ReadoutSemantics,
    UpdateRule,
    run_paradigm,
    standard_registry,
)
from kl_projection_message_passing.signals import SignalChannel


def test_standard_registry_separates_established_and_interface_claims() -> None:
    registry = standard_registry()
    assert registry.get("backpropagation").claim_level is ClaimLevel.ESTABLISHED
    assert registry.get("finite-projection-learning").claim_level is ClaimLevel.COROLLARY
    assert registry.get("predictive-coding").claim_level is ClaimLevel.INTERFACE_ONLY
    assert registry.get("forward-only").references
    assert len(registry.by_claim_level(ClaimLevel.ESTABLISHED)) >= 3
    with pytest.raises(KeyError, match="unknown paradigm"):
        registry.get("missing")


def test_user_defined_paradigm_runs_through_shared_signal_study() -> None:
    specification = ParadigmSpec(
        "contractive-target",
        "Contractive target transport",
        LocalRelation.USER_DEFINED,
        ApproximationFamily.USER_DEFINED,
        Geometry.USER_DEFINED,
        UpdateRule.USER_DEFINED,
        ReadoutSemantics.USER_DEFINED,
        ClaimLevel.DIAGNOSTIC,
        "Synthetic test module.",
    )
    program = ExecutableParadigm(
        specification,
        transition=lambda value: 0.5 * value,
        readout=lambda value: value,
        signal_channel=SignalChannel.LAYERWISE_TARGET,
    )
    result = run_paradigm(
        program,
        jnp.asarray(8.0),
        steps=3,
        analytic_local_bounds=(0.5, 0.5, 0.5),
    )
    np.testing.assert_allclose(np.asarray(result.signals.norms), [8.0, 4.0, 2.0, 1.0])
    assert result.signals.certified_exponential


def test_registry_is_immutable_and_rejects_duplicates() -> None:
    registry = ParadigmRegistry(())
    entry = ParadigmSpec(
        "custom",
        "Custom",
        LocalRelation.USER_DEFINED,
        ApproximationFamily.USER_DEFINED,
        Geometry.USER_DEFINED,
        UpdateRule.USER_DEFINED,
        ReadoutSemantics.USER_DEFINED,
        ClaimLevel.INTERFACE_ONLY,
        "User supplied.",
    )
    extended = registry.add(entry)
    assert not registry.entries
    assert extended.get("custom") is entry
    with pytest.raises(ValueError, match="unique"):
        ParadigmRegistry((entry, entry))
    with pytest.raises(ValueError, match="scope note"):
        ParadigmSpec(
            "bad",
            "Bad",
            LocalRelation.USER_DEFINED,
            ApproximationFamily.USER_DEFINED,
            Geometry.USER_DEFINED,
            UpdateRule.USER_DEFINED,
            ReadoutSemantics.USER_DEFINED,
            ClaimLevel.INTERFACE_ONLY,
            "",
        )

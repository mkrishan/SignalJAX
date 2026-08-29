# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "deterministic_readout.py",
        "gaussian_softening.py",
        "paradigm_calculus.py",
        "projection_differential_bridge.py",
        "signal_attenuation.py",
        "spn_marginals.py",
    ],
)
def test_focused_example_executes(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip()


def test_reproduction_report_has_machine_precision_errors() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "reproduce_paper_identities.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["consensus_residual"] >= 0
    assert report["factorization_residual"] >= 0
    assert report["deterministic_readout_error"] < 1e-12
    assert report["cyclic_projection_corollary_error"] < 1e-12
    assert report["finite_differential_error"] < 1e-5
    assert report["factor_graph_marginal_error"] < 1e-12
    assert report["spn_marginal_error"] < 1e-12
    assert report["unfolding_merge_error"] < 1e-12
    assert report["toy_signal_certified_exponential"]

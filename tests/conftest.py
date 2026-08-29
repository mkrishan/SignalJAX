# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax


def pytest_sessionstart() -> None:
    jax.config.update("jax_enable_x64", True)

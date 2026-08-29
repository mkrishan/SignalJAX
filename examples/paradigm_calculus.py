# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""Inspect the built-in paradigm map and its explicit claim levels."""

from __future__ import annotations

import json
from collections import Counter

from kl_projection_message_passing import standard_registry


def main() -> None:
    registry = standard_registry()
    claim_counts = Counter(entry.claim_level.value for entry in registry.entries)
    report = {
        "claim_counts": dict(sorted(claim_counts.items())),
        "paradigms": {
            entry.identifier: {
                "claim": entry.claim_level.value,
                "readout": entry.readout.value,
                "update": entry.update_rule.value,
            }
            for entry in registry.entries
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

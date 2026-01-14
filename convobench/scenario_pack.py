"""Scenario pack utilities.

A ScenarioPack is a versioned bundle of scenarios plus AFVariables settings.
This module provides helpers to instantiate scenarios from a pack.
"""

from __future__ import annotations

from typing import Any

from convobench.scenarios.base import ScenarioRegistry
from convobench.spec import ScenarioPack


def instantiate_scenarios(pack: ScenarioPack) -> list[Any]:
    """Instantiate scenarios from a ScenarioPack."""
    scenarios = []
    for ref in pack.scenarios:
        scenario_cls = ScenarioRegistry.get(ref.scenario_type)
        if scenario_cls is None:
            raise ValueError(f"Unknown scenario type in pack: {ref.scenario_type}")
        scenarios.append(scenario_cls(**ref.params))
    return scenarios

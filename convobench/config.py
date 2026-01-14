"""Canonical configuration models for ConvoBench.

This module unifies configuration across:
- CLI
- API
- Python runner

The goal is stable, versioned configs that can be stored in RunManifest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from convobench.spec import AFVariables, ScenarioPack, ToolConfig


class ScenarioRunConfig(BaseModel):
    scenario_type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class AgentRunConfig(BaseModel):
    model: str = "mock"
    provider: str = "mock"
    temperature: float = 0.2
    max_tokens: int = 4096
    num_agents: int = 3
    role_descriptions: Optional[List[str]] = None


class BenchmarkRunConfig(BaseModel):
    """Canonical config for a single benchmark execution."""

    config_version: str = "0.1"

    runs_per_scenario: int = 5
    evaluate: bool = True

    scenario: Optional[ScenarioRunConfig] = None
    scenario_pack: Optional[ScenarioPack] = None

    agents: AgentRunConfig = Field(default_factory=AgentRunConfig)
    af_variables: AFVariables = Field(default_factory=AFVariables)
    tool_config: ToolConfig = Field(default_factory=ToolConfig)

    def resolved_scenarios(self) -> List[ScenarioRunConfig]:
        if self.scenario_pack is not None:
            return [ScenarioRunConfig(scenario_type=s.scenario_type, params=s.params) for s in self.scenario_pack.scenarios]
        if self.scenario is not None:
            return [self.scenario]
        raise ValueError("BenchmarkRunConfig requires either scenario or scenario_pack")

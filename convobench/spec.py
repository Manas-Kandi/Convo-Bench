"""Benchmark specification primitives for Agent Factors (AF) research.

This module defines:
- AFVariables: a structured schema for independent variables (what we manipulate)
- ScenarioPack: a versioned, reusable bundle of scenarios + variable settings
- RunManifest: a versioned, reproducible record of how a benchmark run was configured

These are intended to be stable, versioned, and serializable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


AFSchemaVersion = Literal["0.1"]


class EquipmentVariables(BaseModel):
    """Agent-facing interface/tooling variables."""

    tool_schema_strictness: Literal["none", "loose", "strict"] = "strict"
    memory_provenance: Literal["none", "timestamps", "source_ids", "full"] = "timestamps"
    handoff_template: Literal["freeform", "structured", "checksum_readback"] = "structured"
    allowed_message_formats: List[Literal["plain", "json", "markdown"]] = Field(
        default_factory=lambda: ["plain", "json"]
    )


class TaskDesignVariables(BaseModel):
    """Workflow design variables."""

    chain_topology: Literal["chain", "round_robin", "broadcast", "hierarchical"] = "chain"
    redundancy: Literal["none", "dual", "triple"] = "none"
    verification_checkpoints: List[Literal["none", "per_step", "final", "random"]] = Field(
        default_factory=lambda: ["final"]
    )


class EnvironmentVariables(BaseModel):
    """Operational environment variables."""

    noise_level: Literal["none", "low", "medium", "high"] = "none"
    partial_observability: bool = False
    time_pressure: Literal["none", "soft", "hard"] = "none"
    tool_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_latency_ms: int = Field(default=0, ge=0)


class TrainingVariables(BaseModel):
    """Optional training/protocol shaping variables."""

    protocol_prompt: Optional[str] = None
    curriculum_stage: Optional[Literal["intro", "intermediate", "advanced"]] = None


class SelectionVariables(BaseModel):
    """Agent selection + composition variables."""

    model_mix: Literal["single", "heterogeneous"] = "single"
    role_assignment: Literal["none", "static", "dynamic"] = "static"


class AFVariables(BaseModel):
    """Top-level AF independent variable schema."""

    schema_version: AFSchemaVersion = "0.1"
    equipment: EquipmentVariables = Field(default_factory=EquipmentVariables)
    task: TaskDesignVariables = Field(default_factory=TaskDesignVariables)
    environment: EnvironmentVariables = Field(default_factory=EnvironmentVariables)
    training: TrainingVariables = Field(default_factory=TrainingVariables)
    selection: SelectionVariables = Field(default_factory=SelectionVariables)


class ToolConfig(BaseModel):
    """Tool/environment configuration for a run."""

    enabled_tools: List[str] = Field(default_factory=list)
    strict_mode: bool = True
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = Field(default=0, ge=0)


class ScenarioRef(BaseModel):
    """Reference to a scenario with parameterization."""

    scenario_type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ScenarioPack(BaseModel):
    """A reusable, versioned bundle of scenarios and AF variable settings."""

    pack_id: str
    name: str
    version: str = "0.1.0"
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    af_variables: AFVariables = Field(default_factory=AFVariables)
    scenarios: List[ScenarioRef] = Field(default_factory=list)

    @field_validator("scenarios")
    @classmethod
    def _validate_non_empty(cls, v: List[ScenarioRef]) -> List[ScenarioRef]:
        if len(v) == 0:
            raise ValueError("ScenarioPack.scenarios must contain at least one scenario")
        return v


class RunManifest(BaseModel):
    """A versioned, reproducible record of a benchmark run configuration."""

    manifest_version: Literal["0.1"] = "0.1"
    run_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Reproducibility
    code_version: Optional[str] = None
    scenario_pack_version: Optional[str] = None

    # Execution
    seeds: Dict[str, int] = Field(default_factory=dict)
    model_ids: List[str] = Field(default_factory=list)
    temperatures: List[float] = Field(default_factory=list)
    max_tokens: List[int] = Field(default_factory=list)

    # Spec
    af_variables: AFVariables = Field(default_factory=AFVariables)
    tool_config: ToolConfig = Field(default_factory=ToolConfig)

    # Scenario selection
    scenarios: List[ScenarioRef] = Field(default_factory=list)

    @field_validator("temperatures")
    @classmethod
    def _validate_temps(cls, v: List[float]) -> List[float]:
        for t in v:
            if t < 0.0 or t > 2.0:
                raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("seeds")
    @classmethod
    def _validate_seeds(cls, v: Dict[str, int]) -> Dict[str, int]:
        for k, seed in v.items():
            if seed < 0:
                raise ValueError(f"seed for {k} must be non-negative")
        return v

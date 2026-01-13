"""Base classes for benchmark scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from convobench.core.agent import Agent, AgentChain, AgentConfig
from convobench.core.environment import Environment, ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole


class ScenarioConfig(BaseModel):
    """Configuration for a benchmark scenario."""
    
    name: str = Field(description="Scenario name")
    description: str = Field(description="Scenario description")
    category: str = Field(description="Scenario category (relay, planning, coordination, adversarial)")
    difficulty: str = Field(default="medium", description="Difficulty level (easy, medium, hard)")
    
    chain_length: int = Field(default=3, ge=1, description="Number of agents in chain")
    agent_configs: Optional[list[AgentConfig]] = Field(default=None)
    
    initial_state: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)
    
    expected_outcome: Optional[str] = Field(default=None)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    
    max_steps: int = Field(default=50, gt=0)
    timeout_seconds: float = Field(default=300.0, gt=0)


@dataclass
class ScenarioInstance:
    """A specific instance of a scenario ready for execution."""
    
    scenario_id: str
    config: ScenarioConfig
    initial_message: Message
    environment: ScenarioEnvironment
    agent_chain: Optional[AgentChain] = None
    agents: list[Agent] = field(default_factory=list)
    ground_truth: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Scenario(ABC):
    """
    Abstract base class for benchmark scenarios.
    
    A scenario defines:
    - The initial message/goal
    - Environment setup (tools, state)
    - Agent roles and configurations
    - Success criteria and constraints
    - Ground truth for evaluation
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.config = config or self._default_config()
        self._id = str(uuid4())[:8]
    
    @property
    def scenario_id(self) -> str:
        return f"{self.config.name}_{self._id}"
    
    @abstractmethod
    def _default_config(self) -> ScenarioConfig:
        """Return default configuration for this scenario type."""
        pass
    
    @abstractmethod
    def generate_initial_message(self) -> Message:
        """Generate the initial message/goal for the workflow."""
        pass
    
    @abstractmethod
    def setup_environment(self) -> ScenarioEnvironment:
        """Set up the environment with tools and initial state."""
        pass
    
    @abstractmethod
    def get_ground_truth(self) -> dict[str, Any]:
        """
        Return ground truth for evaluation.
        
        This includes:
        - expected_final_output: What the final agent should produce
        - preserved_information: Key information that must be preserved
        - required_actions: Actions that must be taken
        - constraint_checks: Specific constraints to verify
        """
        pass
    
    def get_agent_roles(self) -> list[str]:
        """Return role descriptions for each agent in the chain."""
        return [f"Agent {i+1} in a {self.config.chain_length}-agent workflow" 
                for i in range(self.config.chain_length)]
    
    def create_instance(self) -> ScenarioInstance:
        """Create a runnable instance of this scenario."""
        return ScenarioInstance(
            scenario_id=self.scenario_id,
            config=self.config,
            initial_message=self.generate_initial_message(),
            environment=self.setup_environment(),
            ground_truth=self.get_ground_truth(),
            metadata={
                "category": self.config.category,
                "difficulty": self.config.difficulty,
            },
        )
    
    def validate_config(self) -> list[str]:
        """Validate scenario configuration. Returns list of errors."""
        errors = []
        if self.config.chain_length < 1:
            errors.append("chain_length must be at least 1")
        if self.config.max_steps < self.config.chain_length:
            errors.append("max_steps should be >= chain_length")
        return errors


class ScenarioRegistry:
    """Registry for available benchmark scenarios."""
    
    _scenarios: dict[str, type[Scenario]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a scenario class."""
        def decorator(scenario_cls: type[Scenario]):
            cls._scenarios[name] = scenario_cls
            return scenario_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[type[Scenario]]:
        """Get a scenario class by name."""
        return cls._scenarios.get(name)
    
    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered scenario names."""
        return list(cls._scenarios.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Scenario:
        """Create a scenario instance by name."""
        scenario_cls = cls._scenarios.get(name)
        if scenario_cls is None:
            raise ValueError(f"Unknown scenario: {name}")
        return scenario_cls(**kwargs)

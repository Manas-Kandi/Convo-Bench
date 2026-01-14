"""Benchmark scenarios for ConvoBench."""

from convobench.scenarios.base import Scenario, ScenarioConfig
from convobench.scenarios.relay import (
    InformationRelay,
    ConstrainedRelay,
    NoisyRelay,
)
from convobench.scenarios.planning import (
    CollaborativePlanning,
    GoalDecomposition,
    ResourceAllocation,
)
from convobench.scenarios.coordination import (
    ToolCoordination,
    StateSynchronization,
    HandoffScenario,
)
from convobench.scenarios.adversarial import (
    AdversarialRelay,
    ConstraintViolation,
    ErrorInjection,
)
from convobench.scenarios.realism import (
    PartialObservabilityReconciliation,
    InterruptDrivenPlanning,
    ProtocolHandoffExperiment,
    FailureModeSuite,
)

__all__ = [
    "Scenario",
    "ScenarioConfig",
    "InformationRelay",
    "ConstrainedRelay",
    "NoisyRelay",
    "CollaborativePlanning",
    "GoalDecomposition",
    "ResourceAllocation",
    "ToolCoordination",
    "StateSynchronization",
    "HandoffScenario",
    "AdversarialRelay",
    "ConstraintViolation",
    "ErrorInjection",
    "PartialObservabilityReconciliation",
    "InterruptDrivenPlanning",
    "ProtocolHandoffExperiment",
    "FailureModeSuite",
]

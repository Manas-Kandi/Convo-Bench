"""ConvoBench: Multi-Agent Collaboration Benchmark."""

from convobench.core.engine import WorkflowEngine
from convobench.core.agent import Agent, AgentConfig
from convobench.core.environment import Environment
from convobench.core.metrics import MetricsCollector
from convobench.evaluation.evaluator import ExternalEvaluator
from convobench.bench import ConvoBench

__version__ = "0.1.0"
__all__ = [
    "ConvoBench",
    "WorkflowEngine",
    "Agent",
    "AgentConfig",
    "Environment",
    "MetricsCollector",
    "ExternalEvaluator",
]

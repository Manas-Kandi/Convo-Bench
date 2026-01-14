"""ConvoBench: Multi-Agent Collaboration Benchmark Framework."""

from convobench.bench import ConvoBench, BenchmarkConfig, BenchmarkResult
from convobench.core.agent import Agent, MockAgent
from convobench.evaluation.evaluator import ExternalEvaluator
from convobench.spec import AFVariables, RunManifest, ScenarioPack
from convobench.store import RunStore
from convobench import baselines, reporting, leaderboard

__version__ = "0.1.0"

__all__ = [
    "ConvoBench",
    "BenchmarkConfig",
    "BenchmarkResult",
    "Agent",
    "MockAgent",
    "ExternalEvaluator",
    "AFVariables",
    "RunManifest",
    "ScenarioPack",
    "baselines",
    "reporting",
    "leaderboard",
    "RunStore",
]

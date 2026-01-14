"""Comprehensive benchmark sweep runner.

Runs all permutations of:
- Scenarios (configurable subset or all)
- Agent stacks (baselines + custom)
- Models (mock, nvidia, etc.)

Results are stored in RunStore and can be viewed in the frontend.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from convobench.baselines import (
    baseline_extractor_verifier,
    baseline_hierarchical_coordinator,
    baseline_simple_relay,
    baseline_structured_handoff,
)
from convobench.bench import BenchmarkConfig, ConvoBench, create_mock_agents
from convobench.core.agent import Agent
from convobench.scenarios import (
    ConstrainedRelay,
    InformationRelay,
    NoisyRelay,
    PartialObservabilityReconciliation,
    ProtocolHandoffExperiment,
)
from convobench.scenarios.base import Scenario
from convobench.store import RunStore


DEFAULT_SCENARIOS = [
    ("information_relay", lambda: InformationRelay(chain_length=3, message_complexity="medium")),
    ("constrained_relay", lambda: ConstrainedRelay(chain_length=3, num_constraints=3)),
    ("noisy_relay", lambda: NoisyRelay(chain_length=3, noise_level="medium")),
    ("partial_observability", lambda: PartialObservabilityReconciliation(chain_length=3)),
    ("protocol_handoff", lambda: ProtocolHandoffExperiment(chain_length=3, protocol="structured")),
]

DEFAULT_BASELINES = [
    ("simple_relay", lambda n: baseline_simple_relay(n)),
    ("structured_handoff", lambda n: baseline_structured_handoff(n)),
]

DEFAULT_MODELS = ["mock"]


@dataclass
class SweepConfig:
    scenarios: List[tuple[str, Any]] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    baselines: List[tuple[str, Any]] = field(default_factory=lambda: DEFAULT_BASELINES)
    models: List[str] = field(default_factory=lambda: DEFAULT_MODELS)
    runs_per_combo: int = 1
    seed: int = 42
    verbose: bool = True


@dataclass
class SweepResult:
    sweep_id: str
    started_at: str
    finished_at: Optional[str]
    total_combos: int
    completed: int
    results: List[Dict[str, Any]]


def run_sweep(config: SweepConfig, store: Optional[RunStore] = None) -> SweepResult:
    """Run a comprehensive benchmark sweep synchronously (blocking).

    This is intended to be called from CLI or a background task.
    """
    return asyncio.run(_run_sweep_async(config, store))


async def _run_sweep_async(config: SweepConfig, store: Optional[RunStore] = None) -> SweepResult:
    sweep_id = f"sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    started_at = datetime.utcnow().isoformat()

    combos = list(itertools.product(config.scenarios, config.baselines, config.models))
    total = len(combos)

    if config.verbose:
        print(f"[Sweep {sweep_id}] Starting {total} combinations...")

    results: List[Dict[str, Any]] = []

    for idx, ((scenario_name, scenario_factory), (baseline_name, agent_factory), model) in enumerate(combos):
        combo_label = f"{scenario_name} | {baseline_name} | {model}"
        if config.verbose:
            print(f"  [{idx + 1}/{total}] {combo_label}")

        scenario: Scenario = scenario_factory()
        scenario.set_seed(config.seed + idx)

        num_agents = scenario.config.chain_length
        agents: List[Agent] = agent_factory(num_agents)

        bench = ConvoBench(
            config=BenchmarkConfig(
                runs_per_scenario=config.runs_per_combo,
                save_traces=False,
                verbose=False,
            )
        )

        try:
            result = await bench.run(scenario, agents, runs=config.runs_per_combo)
            summary = {
                "sweep_id": sweep_id,
                "scenario": scenario_name,
                "baseline": baseline_name,
                "model": model,
                "benchmark_id": result.benchmark_id,
                "status": "completed",
                "aggregate_metrics": result.aggregate_metrics.to_dict() if result.aggregate_metrics else None,
                "trace_count": len(result.traces),
            }
        except Exception as e:
            summary = {
                "sweep_id": sweep_id,
                "scenario": scenario_name,
                "baseline": baseline_name,
                "model": model,
                "benchmark_id": None,
                "status": "failed",
                "error": str(e),
            }

        results.append(summary)

        # Persist to store if provided
        if store is not None:
            store.upsert_run(
                run_id=f"{sweep_id}_{idx}",
                created_at=datetime.utcnow().isoformat(),
                status=summary["status"],
                config_json=summary,
                manifest_json=None,
            )

    finished_at = datetime.utcnow().isoformat()

    if config.verbose:
        print(f"[Sweep {sweep_id}] Completed {len(results)}/{total} combinations.")

    return SweepResult(
        sweep_id=sweep_id,
        started_at=started_at,
        finished_at=finished_at,
        total_combos=total,
        completed=len([r for r in results if r["status"] == "completed"]),
        results=results,
    )

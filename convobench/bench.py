"""Main ConvoBench benchmark runner."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from convobench.core.agent import Agent, AgentChain, MockAgent
from convobench.core.engine import WorkflowConfig, WorkflowEngine
from convobench.core.environment import Environment
from convobench.core.metrics import AggregateMetrics, MetricsCollector
from convobench.core.types import WorkflowTrace
from convobench.evaluation.analysis import ComparisonReport, StatisticalAnalyzer, generate_leaderboard
from convobench.evaluation.evaluator import EvaluationResult, EvaluatorConfig, ExternalEvaluator
from convobench.scenarios.base import Scenario, ScenarioInstance
from convobench.spec import RunManifest, ScenarioPack
from convobench.utils.versioning import get_git_commit_hash


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    runs_per_scenario: int = 10
    parallel_runs: bool = False
    max_parallel: int = 5
    save_traces: bool = True
    output_dir: str = "results"
    evaluator_model: str = "gpt-4"
    evaluator_provider: str = "openai"
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run."""
    
    benchmark_id: str
    timestamp: datetime
    config: BenchmarkConfig
    scenarios: list[str]
    configurations: list[str]
    traces: list[WorkflowTrace] = field(default_factory=list)
    evaluations: list[EvaluationResult] = field(default_factory=list)
    aggregate_metrics: Optional[AggregateMetrics] = None
    comparison_report: Optional[ComparisonReport] = None
    run_manifest: Optional[RunManifest] = None
    scenario_pack: Optional[ScenarioPack] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp.isoformat(),
            "scenarios": self.scenarios,
            "configurations": self.configurations,
            "num_traces": len(self.traces),
            "num_evaluations": len(self.evaluations),
            "aggregate_metrics": self.aggregate_metrics.to_dict() if self.aggregate_metrics else None,
            "comparison_report": self.comparison_report.to_dict() if self.comparison_report else None,
            "run_manifest": self.run_manifest.model_dump() if self.run_manifest else None,
            "scenario_pack": self.scenario_pack.model_dump() if self.scenario_pack else None,
        }


class ConvoBench:
    """
    Main benchmark orchestrator for ConvoBench.
    
    Coordinates scenario execution, evaluation, and analysis
    across multiple agent configurations.
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        self.config = config or BenchmarkConfig()
        self.console = Console()
        self.metrics = MetricsCollector()
        self.analyzer = StatisticalAnalyzer()
        self.evaluator = ExternalEvaluator(
            config=EvaluatorConfig(
                model=self.config.evaluator_model,
                provider=self.config.evaluator_provider,
            )
        )
        self._results: list[BenchmarkResult] = []
    
    async def run(
        self,
        scenario: Union[Scenario, list[Scenario]],
        agents: Union[list[Agent], dict[str, list[Agent]]],
        runs: Optional[int] = None,
        *,
        scenario_pack: Optional[ScenarioPack] = None,
        run_manifest: Optional[RunManifest] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark scenarios with specified agents.
        
        Args:
            scenario: Scenario or list of scenarios to run
            agents: Agents to use (list for single config, dict for comparison)
            runs: Number of runs per scenario (overrides config)
            
        Returns:
            Complete benchmark result
        """
        scenarios = [scenario] if isinstance(scenario, Scenario) else scenario
        runs = runs or self.config.runs_per_scenario
        
        # Normalize agents to dict format
        if isinstance(agents, list):
            agent_configs = {"default": agents}
        else:
            agent_configs = agents
        
        benchmark_id = f"bench_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            timestamp=datetime.utcnow(),
            config=self.config,
            scenarios=[s.scenario_id for s in scenarios],
            configurations=list(agent_configs.keys()),
            scenario_pack=scenario_pack,
        )

        if run_manifest is None:
            # Derive a minimal manifest if not provided
            run_manifest = RunManifest(
                run_id=benchmark_id,
                code_version=get_git_commit_hash(),
                scenario_pack_version=scenario_pack.version if scenario_pack else None,
                scenarios=(scenario_pack.scenarios if scenario_pack else []),
            )
        result.run_manifest = run_manifest
        
        if self.config.verbose:
            self.console.print(f"\n[bold blue]Starting ConvoBench[/bold blue]")
            self.console.print(f"  Scenarios: {len(scenarios)}")
            self.console.print(f"  Configurations: {len(agent_configs)}")
            self.console.print(f"  Runs per scenario: {runs}")
            self.console.print()
        
        # Run scenarios
        all_traces: dict[str, list[WorkflowTrace]] = {config: [] for config in agent_configs}
        all_ground_truths: list[dict[str, Any]] = []
        
        # Deterministic seed plan (per scenario run)
        # We derive a stable seed base from the benchmark_id unless a manifest provides one.
        seed_base = int(uuid4().hex[:8], 16)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            disable=not self.config.verbose,
        ) as progress:
            
            for scenario in scenarios:
                task = progress.add_task(f"Running {scenario.config.name}...", total=None)

                # Capture ground truth per run (scenario instances can vary with seed)

                for config_name, config_agents in agent_configs.items():
                    for run_idx in range(runs):
                        progress.update(task, description=f"{scenario.config.name} | {config_name} | Run {run_idx + 1}/{runs}")

                        run_seed = seed_base + run_idx
                        # Apply seed to scenario if supported
                        if hasattr(scenario, "set_seed"):
                            scenario.set_seed(run_seed)

                        instance = scenario.create_instance()
                        all_ground_truths.append(instance.ground_truth)
                        
                        trace = await self._run_single(
                            instance=instance,
                            agents=config_agents,
                            config_name=config_name,
                            seed=run_seed,
                        )
                        
                        all_traces[config_name].append(trace)
                        result.traces.append(trace)
                        
                        # Reset agents for next run
                        for agent in config_agents:
                            agent.reset()
                        instance.environment.reset()
                
                progress.remove_task(task)
        
        # Evaluate results
        if self.config.verbose:
            self.console.print("\n[bold]Evaluating results...[/bold]")
        
        evaluations_by_config: dict[str, list[EvaluationResult]] = {}
        
        for config_name, traces in all_traces.items():
            evaluations = []
            for trace, gt in zip(traces, all_ground_truths * runs):
                eval_result = await self.evaluator.evaluate(trace, gt)
                evaluations.append(eval_result)
                result.evaluations.append(eval_result)
                
                # Update metrics with scores
                self.metrics.update_scores(
                    trace.workflow_id,
                    eval_result.to_metrics_dict(),
                )
            evaluations_by_config[config_name] = evaluations
        
        # Generate analysis
        if len(agent_configs) > 1:
            result.comparison_report = self.analyzer.analyze_results(
                evaluations_by_config,
                scenario_id=scenarios[0].scenario_id if len(scenarios) == 1 else "multiple",
            )
        
        # Aggregate metrics
        result.aggregate_metrics = self.metrics.aggregate()
        
        # Save results
        if self.config.save_traces:
            self._save_results(result)
        
        self._results.append(result)
        
        if self.config.verbose:
            self._print_summary(result)
        
        return result
    
    async def _run_single(
        self,
        instance: ScenarioInstance,
        agents: list[Agent],
        config_name: str,
        *,
        seed: Optional[int] = None,
    ) -> WorkflowTrace:
        """Run a single scenario instance."""
        if seed is not None:
            instance.seed = seed
            instance.metadata["seed"] = seed
        chain = AgentChain(agents, name=config_name)
        
        engine = WorkflowEngine(
            config=WorkflowConfig(
                max_steps=instance.config.max_steps,
                timeout_seconds=instance.config.timeout_seconds,
            ),
            metrics_collector=self.metrics,
        )
        
        trace = await engine.run_chain(
            chain=chain,
            initial_message=instance.initial_message,
            environment=instance.environment,
            scenario_id=instance.scenario_id,
        )
        
        trace.metadata["config_name"] = config_name
        trace.metadata["category"] = instance.config.category
        trace.metadata["scenario_version"] = instance.metadata.get("scenario_version")
        trace.metadata["seed"] = instance.seed
        
        return trace
    
    def _save_results(self, result: BenchmarkResult) -> None:
        """Save benchmark results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_dir / f"{result.benchmark_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save traces
        traces_path = output_dir / f"{result.benchmark_id}_traces.json"
        with open(traces_path, "w") as f:
            json.dump([t.to_dict() for t in result.traces], f, indent=2, default=str)
        
        # Save evaluations
        evals_path = output_dir / f"{result.benchmark_id}_evaluations.json"
        with open(evals_path, "w") as f:
            json.dump([e.to_dict() for e in result.evaluations], f, indent=2, default=str)
    
    def _print_summary(self, result: BenchmarkResult) -> None:
        """Print benchmark summary to console."""
        self.console.print("\n[bold green]Benchmark Complete[/bold green]\n")
        
        # Summary table
        table = Table(title="Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Benchmark ID", result.benchmark_id)
        table.add_row("Total Runs", str(len(result.traces)))
        table.add_row("Scenarios", str(len(result.scenarios)))
        table.add_row("Configurations", str(len(result.configurations)))
        
        if result.aggregate_metrics:
            table.add_row("Success Rate", f"{result.aggregate_metrics.success_rate:.1%}")
            table.add_row("Avg Duration", f"{result.aggregate_metrics.duration_mean:.0f}ms")
            if result.aggregate_metrics.overall_score_mean:
                table.add_row("Avg Score", f"{result.aggregate_metrics.overall_score_mean:.2f}/5")
        
        self.console.print(table)
        
        # Comparison report
        if result.comparison_report:
            self.console.print("\n[bold]Configuration Rankings[/bold]")
            rankings = result.comparison_report.rankings.get("overall_score", {})
            for config, rank in sorted(rankings.items(), key=lambda x: x[1]):
                self.console.print(f"  {rank}. {config}")
            
            if result.comparison_report.recommendations:
                self.console.print("\n[bold]Recommendations[/bold]")
                for rec in result.comparison_report.recommendations:
                    self.console.print(f"  - {rec}")
    
    def analyze(self, result: Optional[BenchmarkResult] = None) -> ComparisonReport:
        """Analyze benchmark results."""
        result = result or self._results[-1] if self._results else None
        if not result:
            raise ValueError("No results to analyze")
        
        evaluations_by_config: dict[str, list[EvaluationResult]] = {}
        for eval_result in result.evaluations:
            config = next(
                (t.metadata.get("config_name", "default") 
                 for t in result.traces 
                 if t.workflow_id == eval_result.workflow_id),
                "default"
            )
            if config not in evaluations_by_config:
                evaluations_by_config[config] = []
            evaluations_by_config[config].append(eval_result)
        
        return self.analyzer.analyze_results(evaluations_by_config)
    
    def export_report(
        self,
        path: str,
        result: Optional[BenchmarkResult] = None,
        format: str = "json",
    ) -> None:
        """Export benchmark report to file."""
        result = result or self._results[-1] if self._results else None
        if not result:
            raise ValueError("No results to export")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def leaderboard(
        self,
        result: Optional[BenchmarkResult] = None,
        metric: str = "overall_score",
    ) -> list[dict[str, Any]]:
        """Generate leaderboard from results."""
        result = result or self._results[-1] if self._results else None
        if not result:
            raise ValueError("No results for leaderboard")
        
        evaluations_by_config: dict[str, list[EvaluationResult]] = {}
        for eval_result in result.evaluations:
            config = next(
                (t.metadata.get("config_name", "default") 
                 for t in result.traces 
                 if t.workflow_id == eval_result.workflow_id),
                "default"
            )
            if config not in evaluations_by_config:
                evaluations_by_config[config] = []
            evaluations_by_config[config].append(eval_result)
        
        return generate_leaderboard(evaluations_by_config, metric)


def create_mock_agents(
    num_agents: int,
    responses: Optional[list[str]] = None,
) -> list[Agent]:
    """Create mock agents for testing."""
    return [
        MockAgent(
            agent_id=f"mock_agent_{i}",
            responses=responses,
        )
        for i in range(num_agents)
    ]

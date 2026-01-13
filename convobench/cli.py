"""Command-line interface for ConvoBench."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="convobench",
    help="ConvoBench: Multi-Agent Collaboration Benchmark",
)
console = Console()


@app.command()
def run(
    scenario: str = typer.Argument(..., help="Scenario name or 'all'"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    runs: int = typer.Option(10, "--runs", "-n", help="Number of runs per scenario"),
    output: str = typer.Option("results", "--output", "-o", help="Output directory"),
    model: str = typer.Option("gpt-4", "--model", "-m", help="Model to use for agents"),
    evaluator: str = typer.Option("gpt-4", "--evaluator", "-e", help="Model for evaluation"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
):
    """Run benchmark scenarios."""
    from convobench import ConvoBench
    from convobench.bench import BenchmarkConfig, create_mock_agents
    from convobench.scenarios import (
        InformationRelay,
        ConstrainedRelay,
        CollaborativePlanning,
        ToolCoordination,
    )
    
    bench_config = BenchmarkConfig(
        runs_per_scenario=runs,
        output_dir=output,
        evaluator_model=evaluator,
        verbose=verbose,
    )
    
    bench = ConvoBench(config=bench_config)
    
    # Select scenarios
    scenario_map = {
        "relay": InformationRelay,
        "constrained_relay": ConstrainedRelay,
        "planning": CollaborativePlanning,
        "coordination": ToolCoordination,
    }
    
    if scenario == "all":
        scenarios = [cls() for cls in scenario_map.values()]
    elif scenario in scenario_map:
        scenarios = [scenario_map[scenario]()]
    else:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        console.print(f"Available: {', '.join(scenario_map.keys())}, all")
        raise typer.Exit(1)
    
    # Create agents (mock for now, real adapters need API keys)
    agents = create_mock_agents(3)
    
    console.print(f"\n[bold]Running ConvoBench[/bold]")
    console.print(f"Scenario: {scenario}")
    console.print(f"Runs: {runs}")
    console.print(f"Output: {output}\n")
    
    result = asyncio.run(bench.run(scenarios, agents, runs))
    
    console.print(f"\n[green]Results saved to {output}/[/green]")


@app.command()
def list_scenarios():
    """List available benchmark scenarios."""
    from convobench.scenarios.base import ScenarioRegistry
    
    table = Table(title="Available Scenarios")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description")
    
    scenarios = [
        ("information_relay", "relay", "Pass information through agent chains"),
        ("constrained_relay", "relay", "Relay with constraint preservation"),
        ("noisy_relay", "relay", "Relay with noise filtering"),
        ("collaborative_planning", "planning", "Multi-agent collaborative planning"),
        ("goal_decomposition", "planning", "Break down goals into subtasks"),
        ("resource_allocation", "planning", "Allocate limited resources"),
        ("tool_coordination", "coordination", "Coordinate tool usage"),
        ("state_synchronization", "coordination", "Maintain shared state"),
        ("handoff_scenario", "coordination", "Task handoffs between agents"),
        ("adversarial_relay", "adversarial", "Detect adversarial modifications"),
        ("constraint_violation", "adversarial", "Test constraint adherence"),
        ("error_injection", "adversarial", "Handle injected errors"),
    ]
    
    for name, category, desc in scenarios:
        table.add_row(name, category, desc)
    
    console.print(table)


@app.command()
def analyze(
    results_path: str = typer.Argument(..., help="Path to results JSON file"),
    metric: str = typer.Option("overall_score", "--metric", "-m", help="Metric to analyze"),
):
    """Analyze benchmark results."""
    path = Path(results_path)
    if not path.exists():
        console.print(f"[red]File not found: {results_path}[/red]")
        raise typer.Exit(1)
    
    with open(path) as f:
        data = json.load(f)
    
    console.print(f"\n[bold]Analysis: {path.name}[/bold]\n")
    
    if "aggregate_metrics" in data and data["aggregate_metrics"]:
        metrics = data["aggregate_metrics"]
        
        table = Table(title="Aggregate Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Run Count", str(metrics.get("run_count", "N/A")))
        table.add_row("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        
        if "duration" in metrics:
            table.add_row("Avg Duration", f"{metrics['duration'].get('mean', 0):.0f}ms")
        
        if "scores" in metrics and metrics["scores"].get("overall", {}).get("mean"):
            table.add_row("Avg Score", f"{metrics['scores']['overall']['mean']:.2f}/5")
        
        console.print(table)
    
    if "comparison_report" in data and data["comparison_report"]:
        report = data["comparison_report"]
        
        if "rankings" in report and metric in report["rankings"]:
            console.print(f"\n[bold]Rankings ({metric})[/bold]")
            for config, rank in sorted(report["rankings"][metric].items(), key=lambda x: x[1]):
                console.print(f"  {rank}. {config}")
        
        if "recommendations" in report:
            console.print("\n[bold]Recommendations[/bold]")
            for rec in report["recommendations"]:
                console.print(f"  - {rec}")


@app.command()
def compare(
    results_a: str = typer.Argument(..., help="First results file"),
    results_b: str = typer.Argument(..., help="Second results file"),
):
    """Compare two benchmark results."""
    path_a = Path(results_a)
    path_b = Path(results_b)
    
    if not path_a.exists() or not path_b.exists():
        console.print("[red]One or both files not found[/red]")
        raise typer.Exit(1)
    
    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)
    
    console.print(f"\n[bold]Comparison[/bold]")
    console.print(f"  A: {path_a.name}")
    console.print(f"  B: {path_b.name}\n")
    
    table = Table(title="Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("A", style="green")
    table.add_column("B", style="yellow")
    table.add_column("Diff", style="magenta")
    
    metrics_a = data_a.get("aggregate_metrics", {})
    metrics_b = data_b.get("aggregate_metrics", {})
    
    if metrics_a and metrics_b:
        score_a = metrics_a.get("scores", {}).get("overall", {}).get("mean", 0)
        score_b = metrics_b.get("scores", {}).get("overall", {}).get("mean", 0)
        diff = score_a - score_b
        table.add_row(
            "Overall Score",
            f"{score_a:.2f}",
            f"{score_b:.2f}",
            f"{diff:+.2f}",
        )
        
        dur_a = metrics_a.get("duration", {}).get("mean", 0)
        dur_b = metrics_b.get("duration", {}).get("mean", 0)
        table.add_row(
            "Avg Duration",
            f"{dur_a:.0f}ms",
            f"{dur_b:.0f}ms",
            f"{dur_a - dur_b:+.0f}ms",
        )
    
    console.print(table)


@app.command()
def version():
    """Show version information."""
    from convobench import __version__
    console.print(f"ConvoBench v{__version__}")


if __name__ == "__main__":
    app()

"""Example benchmark using NVIDIA NIM models.

This example demonstrates how to run ConvoBench with NVIDIA's models
via their OpenAI-compatible API.

Required: Set NVIDIA_API_KEY environment variable before running.
"""

import asyncio
import os

from convobench import ConvoBench
from convobench.bench import BenchmarkConfig
from convobench.scenarios import (
    InformationRelay,
    ConstrainedRelay,
    CollaborativePlanning,
    ToolCoordination,
)
from convobench.adapters import (
    NVIDIAAdapter,
    NVIDIA_MODELS,
    create_nvidia_agents,
    create_mixed_nvidia_agents,
)


async def run_single_model_benchmark():
    """Run benchmark with a single NVIDIA model."""
    
    print("\n" + "=" * 60)
    print("Single Model Benchmark: Nemotron Nano")
    print("=" * 60)
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=5,
            output_dir="results/nvidia",
            verbose=True,
        )
    )
    
    # Create agents using Nemotron (supports reasoning)
    agents = create_nvidia_agents(
        num_agents=3,
        model="nemotron-nano",
        role_descriptions=[
            "You are the first agent. Pass information accurately to the next agent.",
            "You are the middle agent. Preserve all details from the previous agent.",
            "You are the final agent. Compile and present the complete information.",
        ],
    )
    
    scenario = InformationRelay(chain_length=3, message_complexity="medium")
    
    result = await bench.run(scenario=scenario, agents=agents, runs=5)
    
    print(f"\nCompleted with {len(result.traces)} traces")
    return result


async def run_model_comparison():
    """Compare different NVIDIA models on the same scenario."""
    
    print("\n" + "=" * 60)
    print("Model Comparison Benchmark")
    print("=" * 60)
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=5,
            output_dir="results/nvidia_comparison",
            verbose=True,
        )
    )
    
    scenario = ConstrainedRelay(chain_length=3, num_constraints=3)
    
    # Compare different models
    models_to_compare = {
        "nemotron-nano": create_nvidia_agents(3, model="nemotron-nano"),
        "mistral-large": create_nvidia_agents(3, model="mistral-large"),
        "qwen3-next": create_nvidia_agents(3, model="qwen3-next"),
        "minimax-m2": create_nvidia_agents(3, model="minimax-m2"),
    }
    
    result = await bench.run(
        scenario=scenario,
        agents=models_to_compare,
        runs=5,
    )
    
    # Print comparison
    if result.comparison_report:
        print("\n=== Model Rankings ===")
        rankings = result.comparison_report.rankings.get("overall_score", {})
        for config, rank in sorted(rankings.items(), key=lambda x: x[1]):
            print(f"  {rank}. {config}")
    
    return result


async def run_thinking_models_benchmark():
    """Benchmark models with reasoning/thinking capabilities."""
    
    print("\n" + "=" * 60)
    print("Thinking Models Benchmark")
    print("=" * 60)
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=5,
            output_dir="results/nvidia_thinking",
            verbose=True,
        )
    )
    
    # Use a planning scenario that benefits from reasoning
    scenario = CollaborativePlanning(num_agents=3, planning_domain="project")
    
    # Compare thinking-enabled vs standard models
    agent_configs = {
        "nemotron-thinking": create_nvidia_agents(3, model="nemotron-nano"),
        "kimi-k2-thinking": create_nvidia_agents(3, model="kimi-k2"),
        "mistral-standard": create_nvidia_agents(3, model="mistral-large"),
    }
    
    result = await bench.run(
        scenario=scenario,
        agents=agent_configs,
        runs=5,
    )
    
    return result


async def run_mixed_chain_benchmark():
    """Run benchmark with different models in the same chain."""
    
    print("\n" + "=" * 60)
    print("Mixed Model Chain Benchmark")
    print("=" * 60)
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=5,
            output_dir="results/nvidia_mixed",
            verbose=True,
        )
    )
    
    scenario = ToolCoordination(num_agents=4, task_type="data_pipeline")
    
    # Create a chain with different models for different roles
    mixed_agents = create_mixed_nvidia_agents(
        models=["nemotron-nano", "mistral-large", "qwen3-next", "minimax-m2"],
        role_descriptions=[
            "Extract data from the source system.",
            "Transform and clean the extracted data.",
            "Validate the transformed data.",
            "Load the data into the destination.",
        ],
    )
    
    result = await bench.run(scenario=scenario, agents=mixed_agents, runs=5)
    
    return result


async def run_full_benchmark_suite():
    """Run comprehensive benchmark across all NVIDIA models."""
    
    print("\n" + "=" * 60)
    print("Full NVIDIA Benchmark Suite")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("NVIDIA_API_KEY"):
        print("\nERROR: NVIDIA_API_KEY environment variable not set!")
        print("Please set it before running: export NVIDIA_API_KEY='your-key-here'")
        return None
    
    print("\nAvailable NVIDIA models:")
    for shortname, config in NVIDIA_MODELS.items():
        reasoning = " (reasoning)" if config["supports_reasoning"] else ""
        print(f"  - {shortname}: {config['model_id']}{reasoning}")
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=10,
            output_dir="results/nvidia_full",
            verbose=True,
        )
    )
    
    # Multiple scenarios
    scenarios = [
        InformationRelay(chain_length=4, message_complexity="complex"),
        ConstrainedRelay(chain_length=4, num_constraints=4),
        CollaborativePlanning(num_agents=3, planning_domain="project"),
    ]
    
    # All models
    all_model_configs = {
        shortname: create_nvidia_agents(4, model=shortname)
        for shortname in NVIDIA_MODELS.keys()
    }
    
    result = await bench.run(
        scenario=scenarios,
        agents=all_model_configs,
        runs=10,
    )
    
    # Export detailed report
    bench.export_report("results/nvidia_full/benchmark_report.json", result)
    
    # Print leaderboard
    leaderboard = bench.leaderboard(result)
    print("\n=== Final Leaderboard ===")
    for entry in leaderboard:
        print(f"  {entry['rank']}. {entry['configuration']}: {entry['mean_score']:.2f} (n={entry['n_runs']})")
    
    return result


if __name__ == "__main__":
    # Check for API key first
    if not os.environ.get("NVIDIA_API_KEY"):
        print("=" * 60)
        print("NVIDIA API Key Required")
        print("=" * 60)
        print("\nSet your NVIDIA API key:")
        print("  export NVIDIA_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://build.nvidia.com/")
        print("=" * 60)
    else:
        # Run examples
        asyncio.run(run_single_model_benchmark())
        # asyncio.run(run_model_comparison())
        # asyncio.run(run_thinking_models_benchmark())
        # asyncio.run(run_mixed_chain_benchmark())
        # asyncio.run(run_full_benchmark_suite())

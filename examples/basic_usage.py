"""Basic usage example for ConvoBench."""

import asyncio

from convobench import ConvoBench
from convobench.bench import BenchmarkConfig, create_mock_agents
from convobench.scenarios import InformationRelay, CollaborativePlanning
from convobench.adapters import OpenAIAdapter, AnthropicAdapter


async def run_basic_benchmark():
    """Run a basic benchmark with mock agents."""
    
    # Create benchmark instance
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=5,
            output_dir="results",
            verbose=True,
        )
    )
    
    # Create a simple relay scenario
    scenario = InformationRelay(
        chain_length=3,
        message_complexity="medium",
    )
    
    # Use mock agents for testing (no API keys needed)
    agents = create_mock_agents(3)
    
    # Run the benchmark
    result = await bench.run(
        scenario=scenario,
        agents=agents,
        runs=5,
    )
    
    print(f"\nBenchmark ID: {result.benchmark_id}")
    print(f"Total traces: {len(result.traces)}")
    print(f"Total evaluations: {len(result.evaluations)}")
    
    return result


async def run_comparison_benchmark():
    """Run a benchmark comparing different configurations."""
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=3,
            verbose=True,
        )
    )
    
    scenario = InformationRelay(chain_length=4)
    
    # Compare different mock configurations
    # In real usage, these would be different LLM adapters
    agent_configs = {
        "config_short_responses": create_mock_agents(4, responses=["Brief response"]),
        "config_detailed_responses": create_mock_agents(4, responses=["This is a much more detailed response with additional context and information that might be relevant to the task at hand."]),
    }
    
    result = await bench.run(
        scenario=scenario,
        agents=agent_configs,
        runs=3,
    )
    
    # Print comparison
    if result.comparison_report:
        print("\n=== Comparison Report ===")
        for config, rank in result.comparison_report.rankings.get("overall_score", {}).items():
            print(f"  {rank}. {config}")
    
    return result


async def run_with_real_llms():
    """
    Example of running with real LLM adapters.
    
    Note: Requires API keys to be set in environment variables:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    """
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=10,
            evaluator_model="gpt-4",
            verbose=True,
        )
    )
    
    # Create agents with real LLM adapters
    agents = [
        OpenAIAdapter(
            agent_id="relay_agent_1",
            model="gpt-4",
            role_description="You are the first agent in a relay chain. Pass information accurately.",
        ),
        OpenAIAdapter(
            agent_id="relay_agent_2",
            model="gpt-4",
            role_description="You are a middle agent. Preserve all information from the previous agent.",
        ),
        OpenAIAdapter(
            agent_id="relay_agent_3",
            model="gpt-4",
            role_description="You are the final agent. Compile and present the complete information.",
        ),
    ]
    
    scenario = InformationRelay(chain_length=3, message_complexity="complex")
    
    result = await bench.run(
        scenario=scenario,
        agents=agents,
        runs=10,
    )
    
    # Generate leaderboard
    leaderboard = bench.leaderboard(result)
    print("\n=== Leaderboard ===")
    for entry in leaderboard:
        print(f"  {entry['rank']}. {entry['configuration']}: {entry['mean_score']:.2f}")
    
    return result


async def run_multiple_scenarios():
    """Run multiple scenarios in a single benchmark."""
    
    bench = ConvoBench(
        config=BenchmarkConfig(
            runs_per_scenario=3,
            verbose=True,
        )
    )
    
    # Multiple scenarios
    scenarios = [
        InformationRelay(chain_length=3, message_complexity="simple"),
        InformationRelay(chain_length=5, message_complexity="complex"),
        CollaborativePlanning(num_agents=3, planning_domain="project"),
    ]
    
    agents = create_mock_agents(5)  # Max agents needed
    
    result = await bench.run(
        scenario=scenarios,
        agents=agents,
        runs=3,
    )
    
    print(f"\nCompleted {len(result.scenarios)} scenarios")
    print(f"Total traces: {len(result.traces)}")
    
    return result


if __name__ == "__main__":
    # Run basic example
    print("=" * 50)
    print("Running Basic Benchmark")
    print("=" * 50)
    asyncio.run(run_basic_benchmark())
    
    print("\n" + "=" * 50)
    print("Running Comparison Benchmark")
    print("=" * 50)
    asyncio.run(run_comparison_benchmark())
    
    print("\n" + "=" * 50)
    print("Running Multiple Scenarios")
    print("=" * 50)
    asyncio.run(run_multiple_scenarios())

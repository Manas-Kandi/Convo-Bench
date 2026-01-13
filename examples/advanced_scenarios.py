"""Advanced scenario examples for ConvoBench."""

import asyncio
from typing import Any

from convobench import ConvoBench
from convobench.bench import BenchmarkConfig, create_mock_agents
from convobench.scenarios import (
    AdversarialRelay,
    ConstrainedRelay,
    ErrorInjection,
    GoalDecomposition,
    HandoffScenario,
    NoisyRelay,
    ResourceAllocation,
    StateSynchronization,
    ToolCoordination,
)
from convobench.scenarios.base import Scenario, ScenarioConfig
from convobench.core.environment import ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole


class CustomScenario(Scenario):
    """Example of creating a custom scenario."""
    
    def __init__(self, custom_param: str = "default"):
        self.custom_param = custom_param
        super().__init__()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="custom_scenario",
            description="A custom scenario demonstrating extensibility",
            category="custom",
            difficulty="medium",
            chain_length=3,
            constraints=["Custom constraint 1", "Custom constraint 2"],
        )
    
    def generate_initial_message(self) -> Message:
        return Message(
            role=MessageRole.USER,
            content=f"This is a custom scenario with param: {self.custom_param}",
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        def custom_tool_handler(args, state):
            return {"result": f"Processed: {args.get('input', '')}"}
        
        env = ScenarioEnvironment(
            name="custom_env",
            initial_state={"custom_state": True},
        )
        
        env.register_tool(Tool(
            name="custom_tool",
            description="A custom tool for this scenario",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
            },
            handler=custom_tool_handler,
        ))
        
        return env
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {"custom_param": self.custom_param},
            "preserved_information": {"param": self.custom_param},
            "required_actions": ["custom_tool"],
            "constraint_checks": ["custom_check"],
        }


async def run_adversarial_scenarios():
    """Run adversarial robustness scenarios."""
    
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=5, verbose=True))
    
    scenarios = [
        # Test detection of subtle modifications
        AdversarialRelay(chain_length=5, attack_type="subtle"),
        
        # Test detection of instruction injection
        AdversarialRelay(chain_length=5, attack_type="injection"),
        
        # Test constraint adherence under pressure
        ConstrainedRelay(chain_length=4, num_constraints=5),
        
        # Test error handling and recovery
        ErrorInjection(chain_length=4, error_type="tool_failure"),
    ]
    
    agents = create_mock_agents(5)
    
    for scenario in scenarios:
        print(f"\n--- Running: {scenario.config.name} ---")
        result = await bench.run(scenario=scenario, agents=agents, runs=5)
        
        if result.aggregate_metrics:
            print(f"Success rate: {result.aggregate_metrics.success_rate:.1%}")


async def run_coordination_scenarios():
    """Run tool coordination and state management scenarios."""
    
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=5, verbose=True))
    
    scenarios = [
        # Data pipeline coordination
        ToolCoordination(num_agents=3, task_type="data_pipeline"),
        
        # Deployment workflow
        ToolCoordination(num_agents=4, task_type="deployment"),
        
        # State synchronization
        StateSynchronization(num_agents=3, state_complexity="medium"),
        
        # Task handoffs
        HandoffScenario(num_handoffs=3, task_domain="support"),
    ]
    
    agents = create_mock_agents(4)
    
    for scenario in scenarios:
        print(f"\n--- Running: {scenario.config.name} ---")
        result = await bench.run(scenario=scenario, agents=agents, runs=5)


async def run_planning_scenarios():
    """Run collaborative planning scenarios."""
    
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=5, verbose=True))
    
    scenarios = [
        # Goal decomposition at different complexity levels
        GoalDecomposition(chain_length=4, goal_complexity="simple"),
        GoalDecomposition(chain_length=4, goal_complexity="complex"),
        
        # Resource allocation with different resource types
        ResourceAllocation(num_stakeholders=3, resource_type="budget"),
        ResourceAllocation(num_stakeholders=4, resource_type="personnel"),
    ]
    
    agents = create_mock_agents(5)
    
    for scenario in scenarios:
        print(f"\n--- Running: {scenario.config.name} ---")
        result = await bench.run(scenario=scenario, agents=agents, runs=5)


async def run_noise_filtering_study():
    """Study how well agents filter noise at different levels."""
    
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=10, verbose=True))
    
    results = {}
    
    for noise_level in ["low", "medium", "high"]:
        scenario = NoisyRelay(chain_length=5, noise_level=noise_level)
        agents = create_mock_agents(5)
        
        result = await bench.run(scenario=scenario, agents=agents, runs=10)
        
        if result.aggregate_metrics:
            results[noise_level] = {
                "success_rate": result.aggregate_metrics.success_rate,
                "avg_score": result.aggregate_metrics.overall_score_mean,
            }
    
    print("\n=== Noise Filtering Study Results ===")
    for level, metrics in results.items():
        print(f"  {level}: success={metrics['success_rate']:.1%}, score={metrics.get('avg_score', 'N/A')}")


async def run_chain_length_study():
    """Study how performance degrades with chain length."""
    
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=10, verbose=True))
    
    results = {}
    
    for chain_length in [2, 3, 5, 7, 10]:
        scenario = ConstrainedRelay(chain_length=chain_length, num_constraints=3)
        agents = create_mock_agents(chain_length)
        
        result = await bench.run(scenario=scenario, agents=agents, runs=10)
        
        if result.aggregate_metrics:
            results[chain_length] = {
                "success_rate": result.aggregate_metrics.success_rate,
                "avg_duration": result.aggregate_metrics.duration_mean,
            }
    
    print("\n=== Chain Length Study Results ===")
    for length, metrics in results.items():
        print(f"  Length {length}: success={metrics['success_rate']:.1%}, duration={metrics['avg_duration']:.0f}ms")


async def run_custom_scenario():
    """Run a custom scenario."""
    
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=5, verbose=True))
    
    scenario = CustomScenario(custom_param="test_value")
    agents = create_mock_agents(3)
    
    result = await bench.run(scenario=scenario, agents=agents, runs=5)
    
    print(f"\nCustom scenario completed with {len(result.traces)} traces")


if __name__ == "__main__":
    print("=" * 60)
    print("ConvoBench Advanced Scenarios")
    print("=" * 60)
    
    # Run different scenario types
    asyncio.run(run_adversarial_scenarios())
    asyncio.run(run_coordination_scenarios())
    asyncio.run(run_planning_scenarios())
    
    # Run studies
    asyncio.run(run_noise_filtering_study())
    asyncio.run(run_chain_length_study())
    
    # Run custom scenario
    asyncio.run(run_custom_scenario())

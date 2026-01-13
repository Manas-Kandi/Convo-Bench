# ConvoBench: Multi-Agent Collaboration Benchmark

A systematic benchmark for evaluating AI agent collaboration, information preservation, and workflow execution across multi-agent chains.

## Overview

ConvoBench addresses a critical challenge in autonomous AI systems: **information degradation in multi-agent workflows**. Like the telephone game, when agents pass information through chains, intent drifts, constraints are lost, and meaning mutates. This benchmark provides tools to measure and analyze these breakdowns.

## Key Features

- **Realistic Agentic Scenarios**: Agents operate as text–action–feedback systems with tools, environment changes, and state updates
- **Statistical Analysis**: Run scenarios multiple times to gather statistically significant data
- **External LLM Evaluator**: Independent evaluation outside the agent chain for unbiased assessment
- **Comprehensive Metrics**: Intent preservation, constraint adherence, action correctness, coordination quality, error propagation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ConvoBench Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Scenario   │───▶│   Workflow   │───▶│   Metrics    │       │
│  │   Library    │    │   Engine     │    │   Collector  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Agent     │    │  Environment │    │   External   │       │
│  │   Adapters   │    │   Simulator  │    │   Evaluator  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Scenario Categories

1. **Information Relay** - Pure message passing through agent chains
2. **Collaborative Planning** - Multi-agent goal decomposition and planning
3. **Tool Coordination** - Agents sharing tools and coordinating actions
4. **State Synchronization** - Maintaining shared state across agents
5. **Error Recovery** - Handling failures and propagating corrections
6. **Adversarial Robustness** - Resilience to malformed or misleading inputs

## Installation

```bash
pip install -e .
```

## Supported LLM Providers

### NVIDIA NIM (Primary)

ConvoBench supports NVIDIA NIM models via their OpenAI-compatible API:

| Shortname | Model ID | Features |
|-----------|----------|----------|
| `nemotron-nano` | nvidia/nemotron-3-nano-30b-a3b | Reasoning/Thinking |
| `kimi-k2` | moonshotai/kimi-k2-thinking | Reasoning/Thinking |
| `mistral-large` | mistralai/mistral-large-3-675b-instruct-2512 | High capacity |
| `minimax-m2` | minimaxai/minimax-m2 | Balanced |
| `qwen3-next` | qwen/qwen3-next-80b-a3b-instruct | Instruction-tuned |
| `falcon3` | tiiuae/falcon3-7b-instruct | Lightweight |

```python
from convobench.adapters import create_nvidia_agents, NVIDIAAdapter

# Set your API key
# export NVIDIA_API_KEY='your-key-here'

# Create agents with a specific model
agents = create_nvidia_agents(3, model="nemotron-nano")

# Or create individual agents
agent = NVIDIAAdapter(
    agent_id="my_agent",
    model="mistral-large",
    role_description="You are a helpful assistant.",
)
```

### Other Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku

## Quick Start

```python
from convobench import ConvoBench, scenarios

# Initialize benchmark
bench = ConvoBench()

# Run a scenario
results = bench.run(
    scenario=scenarios.InformationRelay(chain_length=5),
    agents=["gpt-4", "claude-3", "gemini-pro"],
    runs=100
)

# Analyze results
bench.analyze(results)
bench.export_report("results/relay_benchmark.json")
```

## Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| **Intent Preservation** | Does the final output match the original intent? |
| **Constraint Adherence** | Are specified constraints maintained throughout? |
| **Action Correctness** | Are tool calls and actions appropriate? |
| **Coordination Quality** | How well do agents synchronize? |
| **Error Propagation** | How do errors compound through the chain? |
| **Latency & Efficiency** | Time and token costs of workflows |

## Project Structure

```
convobench/
├── core/                 # Core framework components
│   ├── engine.py         # Workflow execution engine
│   ├── agent.py          # Agent base classes and adapters
│   ├── environment.py    # Simulated environment
│   └── metrics.py        # Metrics collection
├── scenarios/            # Benchmark scenarios
│   ├── base.py           # Base scenario class
│   ├── relay.py          # Information relay scenarios
│   ├── planning.py       # Collaborative planning
│   ├── coordination.py   # Tool coordination
│   └── adversarial.py    # Adversarial scenarios
├── evaluation/           # Evaluation framework
│   ├── evaluator.py      # External LLM evaluator
│   ├── rubrics.py        # Evaluation rubrics
│   └── analysis.py       # Statistical analysis
├── adapters/             # LLM provider adapters
│   ├── openai.py
│   ├── anthropic.py
│   └── base.py
└── utils/                # Utilities
    ├── logging.py
    └── serialization.py
```

## License

MIT License

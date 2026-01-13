# ConvoBench Architecture

This document describes the architecture and design decisions behind ConvoBench.

## Overview

ConvoBench is designed to systematically evaluate multi-agent AI collaboration by simulating realistic agentic workflows and measuring information preservation, coordination quality, and error handling across agent chains.

## Core Design Principles

### 1. Agents as Text-Action-Feedback Systems

Unlike simple chatbot evaluations, ConvoBench treats agents as complete systems that:
- Receive textual inputs
- Take actions (tool calls, delegations)
- Process feedback from the environment
- Update internal state

This reflects how agents actually operate in real-world autonomous scenarios.

### 2. External Evaluation

The evaluation system sits **outside** the agent chain to ensure unbiased assessment. The external evaluator:
- Has no participation in the workflow
- Receives the complete trace after execution
- Compares against ground truth
- Applies standardized rubrics

### 3. Statistical Rigor

Single runs are insufficient for reliable conclusions. ConvoBench supports:
- Multiple runs per scenario
- Statistical aggregation (mean, std, confidence intervals)
- Significance testing for comparisons
- Reliability metrics (coefficient of variation)

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ConvoBench                                     │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Scenarios  │  │   Engine    │  │  Adapters   │  │ Evaluation  │    │
│  │             │  │             │  │             │  │             │    │
│  │ - Relay     │  │ - Chain     │  │ - OpenAI    │  │ - Rubrics   │    │
│  │ - Planning  │  │ - Collab    │  │ - Anthropic │  │ - Evaluator │    │
│  │ - Coord     │  │ - Parallel  │  │ - Custom    │  │ - Analysis  │    │
│  │ - Adversar  │  │             │  │             │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          │                │                              │
│                          ▼                ▼                              │
│                   ┌─────────────┐  ┌─────────────┐                      │
│                   │ Environment │  │   Metrics   │                      │
│                   │             │  │  Collector  │                      │
│                   │ - Tools     │  │             │                      │
│                   │ - State     │  │ - Traces    │                      │
│                   │ - Events    │  │ - Scores    │                      │
│                   └─────────────┘  └─────────────┘                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Scenarios (`convobench/scenarios/`)

Scenarios define the test cases. Each scenario specifies:

| Component | Purpose |
|-----------|---------|
| Initial Message | The starting input/goal |
| Environment | Tools and initial state |
| Ground Truth | Expected outcomes for evaluation |
| Constraints | Rules that must be maintained |
| Agent Roles | Role descriptions for each position |

**Scenario Categories:**

1. **Relay** - Information passing through chains
   - `InformationRelay` - Basic message preservation
   - `ConstrainedRelay` - Preserve constraints alongside content
   - `NoisyRelay` - Filter noise while preserving signal

2. **Planning** - Collaborative goal achievement
   - `CollaborativePlanning` - Multi-agent plan creation
   - `GoalDecomposition` - Break down complex objectives
   - `ResourceAllocation` - Distribute limited resources

3. **Coordination** - Tool and state management
   - `ToolCoordination` - Sequence tool usage
   - `StateSynchronization` - Maintain shared state
   - `HandoffScenario` - Transfer context between agents

4. **Adversarial** - Robustness testing
   - `AdversarialRelay` - Detect malicious modifications
   - `ConstraintViolation` - Resist pressure to violate rules
   - `ErrorInjection` - Handle and recover from errors

### Workflow Engine (`convobench/core/engine.py`)

The engine orchestrates workflow execution with multiple coordination strategies:

**Chain Execution:**
```
Agent 1 → Agent 2 → Agent 3 → ... → Agent N
```
Each agent processes the output of the previous agent.

**Round Robin:**
```
Agent 1 → Agent 2 → Agent 3 → Agent 1 → Agent 2 → ...
```
Agents take turns until termination.

**Broadcast:**
```
        ┌→ Agent 1 ─┐
Message ├→ Agent 2 ─┼→ Aggregated → Next Round
        └→ Agent 3 ─┘
```
All agents receive the same message and responses are aggregated.

**Hierarchical:**
```
Coordinator
    ├── Worker 1
    ├── Worker 2
    └── Worker 3
```
One agent coordinates, others execute delegated tasks.

### Environment (`convobench/core/environment.py`)

Simulates the external world agents interact with:

- **Tools**: Functions agents can call
- **State**: Shared variables and resources
- **Events**: Logged actions for analysis

Tools are defined with:
- Name and description
- Parameter schema (JSON Schema)
- Handler function
- Side effects declaration

### Evaluation (`convobench/evaluation/`)

**Rubrics** define scoring criteria across dimensions:
- Intent Preservation
- Constraint Adherence
- Action Correctness
- Coordination Quality
- Error Propagation
- Information Fidelity

**External Evaluator** uses an LLM to:
1. Analyze the complete workflow trace
2. Compare against ground truth
3. Score each dimension (1-5)
4. Provide evidence and recommendations

**Statistical Analyzer** computes:
- Summary statistics per configuration
- Pairwise comparisons with significance tests
- Effect sizes (Cohen's d)
- Reliability metrics

### Adapters (`convobench/adapters/`)

Adapters connect ConvoBench to LLM providers:

```python
class BaseAdapter(Agent):
    async def process(message, tools) -> (response, actions)
    async def execute_tool(tool_call) -> result
```

Implementations handle provider-specific:
- API authentication
- Message formatting
- Tool schema translation
- Response parsing

## Data Flow

```
1. Scenario creates instance with:
   - Initial message
   - Environment (tools, state)
   - Ground truth

2. Engine executes workflow:
   - For each agent in chain/strategy:
     - Agent processes message
     - Agent may call tools
     - Environment executes tools
     - Results fed back to agent
     - Output passed to next agent

3. Metrics collector records:
   - Complete trace (all steps)
   - Timing information
   - Token counts
   - Tool call results

4. External evaluator assesses:
   - Receives trace + ground truth
   - Applies rubric
   - Scores each dimension
   - Generates recommendations

5. Statistical analyzer aggregates:
   - Across multiple runs
   - Across configurations
   - Generates comparison report
```

## Extensibility

### Custom Scenarios

```python
class MyScenario(Scenario):
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(name="my_scenario", ...)
    
    def generate_initial_message(self) -> Message:
        return Message(role=MessageRole.USER, content="...")
    
    def setup_environment(self) -> ScenarioEnvironment:
        env = ScenarioEnvironment(name="my_env")
        env.register_tool(Tool(...))
        return env
    
    def get_ground_truth(self) -> dict:
        return {"expected": ..., "preserved": ...}
```

### Custom Adapters

```python
class MyAdapter(BaseAdapter):
    async def _initialize_client(self):
        self._client = MyLLMClient()
    
    async def _call_api(self, messages, tools):
        response = await self._client.complete(messages, tools)
        return self._parse_response(response)
```

### Custom Rubrics

```python
rubric = EvaluationRubric(
    name="Custom Evaluation",
    description="My custom rubric",
    dimensions=[
        RubricDimension(
            name="Custom Metric",
            description="What this measures",
            weight=2.0,
            score_descriptions={
                ScoreLevel.EXCELLENT: "Perfect performance",
                # ...
            }
        ),
    ]
)
```

## Performance Considerations

- **Async execution**: All LLM calls are async
- **Parallel runs**: Optional parallel execution of independent runs
- **Trace storage**: Configurable persistence
- **Memory efficiency**: Streaming for large traces

## Future Directions

1. **Real-time monitoring**: Live dashboards during execution
2. **Automated scenario generation**: LLM-generated test cases
3. **Human evaluation integration**: Hybrid human-AI assessment
4. **Cross-benchmark compatibility**: Standard output formats

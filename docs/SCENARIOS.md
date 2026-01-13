# Scenario Reference

This document provides detailed information about all benchmark scenarios in ConvoBench.

## Scenario Categories

ConvoBench organizes scenarios into four categories:

| Category | Focus | Key Challenge |
|----------|-------|---------------|
| **Relay** | Information passing | Preservation through chains |
| **Planning** | Goal achievement | Coordination and decomposition |
| **Coordination** | Tool/state management | Sequencing and synchronization |
| **Adversarial** | Robustness | Detection and recovery |

---

## Relay Scenarios

### InformationRelay

**Purpose**: Test basic information preservation through agent chains.

**Configuration**:
```python
InformationRelay(
    chain_length=5,           # Number of agents
    message_complexity="medium"  # simple, medium, complex
)
```

**Complexity Levels**:
- `simple`: Basic task with few fields (meeting scheduling)
- `medium`: Multi-field task with requirements and constraints
- `complex`: Nested data with milestones, budgets, and metrics

**Ground Truth Checks**:
- All keys preserved
- Numerical values exact
- No hallucinated content
- Structure maintained

**Use When**: Establishing baseline information preservation capabilities.

---

### ConstrainedRelay

**Purpose**: Test constraint preservation alongside content.

**Configuration**:
```python
ConstrainedRelay(
    chain_length=4,
    num_constraints=3
)
```

**Constraint Types**:
- Confidentiality rules
- Timing requirements
- Approval requirements
- Budget limits
- Format specifications
- Security rules
- Compliance requirements
- Communication protocols

**Ground Truth Checks**:
- All constraints mentioned
- Constraint types preserved
- Task details complete

**Use When**: Testing whether agents maintain rules and limitations.

---

### NoisyRelay

**Purpose**: Test ability to filter irrelevant information.

**Configuration**:
```python
NoisyRelay(
    chain_length=5,
    noise_level="medium"  # low, medium, high
)
```

**Noise Levels**:
- `low`: 2 distracting messages
- `medium`: 4 distracting messages
- `high`: 6 distracting messages

**Ground Truth Checks**:
- Core task preserved
- Noise not included
- No confusion between noise and task

**Use When**: Testing signal extraction from noisy inputs.

---

## Planning Scenarios

### CollaborativePlanning

**Purpose**: Test multi-agent collaborative plan creation.

**Configuration**:
```python
CollaborativePlanning(
    num_agents=3,
    planning_domain="project"  # project, event, migration
)
```

**Planning Domains**:
- `project`: Software/product development
- `event`: Conference/event organization
- `migration`: System migration

**Agent Expertise** (auto-assigned):
- Technical Lead / Infrastructure Architect
- Product Manager / Event Coordinator
- Project Manager / Operations Manager

**Environment Tools**:
- `update_shared_plan`: Add/update plan elements
- `get_shared_plan`: Retrieve current plan

**Ground Truth Checks**:
- All agents contributed
- Plan addresses requirements
- No contradictions
- Timeline feasible
- Budget respected

**Use When**: Testing collaborative problem-solving.

---

### GoalDecomposition

**Purpose**: Test breaking down high-level goals into actionable subtasks.

**Configuration**:
```python
GoalDecomposition(
    chain_length=4,
    goal_complexity="medium"  # simple, medium, complex
)
```

**Agent Roles**:
1. Identify major workstreams/phases
2. Break into specific tasks
3. Map dependencies and sequencing
4. Validate and finalize

**Ground Truth Checks**:
- All goal aspects covered
- Subtasks are actionable
- Dependencies logical
- No orphan tasks

**Use When**: Testing systematic problem decomposition.

---

### ResourceAllocation

**Purpose**: Test negotiation and trade-off analysis.

**Configuration**:
```python
ResourceAllocation(
    num_stakeholders=3,
    resource_type="budget"  # budget, personnel, time
)
```

**Resource Types**:
- `budget`: Dollar allocation with min/max constraints
- `personnel`: FTE allocation with team size limits
- `time`: Hour allocation with min/max per project

**Environment Tools**:
- `propose_allocation`: Submit allocation proposal

**Ground Truth Checks**:
- Total within budget
- All stakeholders allocated
- Minimum allocations met
- Allocation justified

**Use When**: Testing consensus building under constraints.

---

## Coordination Scenarios

### ToolCoordination

**Purpose**: Test sequenced tool usage across agents.

**Configuration**:
```python
ToolCoordination(
    num_agents=3,
    task_type="data_pipeline"  # data_pipeline, deployment, document_processing
)
```

**Task Types**:

| Type | Steps |
|------|-------|
| data_pipeline | extract → transform → load |
| deployment | build → test → deploy → verify |
| document_processing | ingest → extract_entities → classify → summarize |

**Ground Truth Checks**:
- Tools called in order
- All tools called
- Outputs passed correctly

**Use When**: Testing workflow orchestration.

---

### StateSynchronization

**Purpose**: Test shared state management.

**Configuration**:
```python
StateSynchronization(
    num_agents=3,
    state_complexity="medium"  # simple, medium, complex
)
```

**Complexity Levels**:
- `simple`: Counter and list operations
- `medium`: Inventory management with orders
- `complex`: Multi-account financial operations

**Environment Tools**:
- `read_state`: Get current state
- `update_state`: Modify state (set, increment, decrement, append)

**Ground Truth Checks**:
- State consistent
- All operations applied
- No data loss

**Use When**: Testing concurrent state management.

---

### HandoffScenario

**Purpose**: Test context transfer between agents.

**Configuration**:
```python
HandoffScenario(
    num_handoffs=3,
    task_domain="support"  # support, investigation, development
)
```

**Task Domains**:
- `support`: Customer support ticket resolution
- `investigation`: Security incident investigation
- `development`: Feature development handoff

**Environment Tools**:
- `log_work`: Record completed work
- `create_handoff`: Create handoff summary

**Handoff Requirements**:
- Completed work documented
- Pending items listed
- Context preserved
- Blockers identified

**Use When**: Testing shift/team transitions.

---

## Adversarial Scenarios

### AdversarialRelay

**Purpose**: Test detection of malicious modifications.

**Configuration**:
```python
AdversarialRelay(
    chain_length=5,
    adversarial_position=None,  # Random if None
    attack_type="subtle"  # subtle, injection, omission, fabrication
)
```

**Attack Types**:

| Type | Description |
|------|-------------|
| subtle | Small value changes (amounts, account numbers) |
| injection | Add malicious instructions |
| omission | Remove critical fields |
| fabrication | Add fake authorizations |

**Environment Tools**:
- `flag_suspicious`: Report detected modifications

**Ground Truth Checks**:
- Adversarial detected
- Original values preserved
- Verification not bypassed

**Use When**: Testing security and integrity.

---

### ConstraintViolation

**Purpose**: Test resistance to pressure for rule violations.

**Configuration**:
```python
ConstraintViolation(
    num_agents=3,
    violation_type="budget"  # budget, authorization, privacy
)
```

**Violation Types**:
- `budget`: Request exceeding spending limits
- `authorization`: Request bypassing approval
- `privacy`: Request violating data rules

**Environment Tools**:
- `check_constraint`: Verify compliance
- `escalate_request`: Escalate to authority
- `reject_request`: Refuse with reason

**Ground Truth Checks**:
- Violation identified
- Appropriate action taken
- Reasoning documented

**Use When**: Testing policy compliance.

---

### ErrorInjection

**Purpose**: Test error handling and recovery.

**Configuration**:
```python
ErrorInjection(
    chain_length=4,
    error_type="tool_failure",  # tool_failure, validation_error, permission_error, data_corruption
    error_position=None  # Random if None
)
```

**Error Types**:

| Type | Recoverable | Recovery Action |
|------|-------------|-----------------|
| tool_failure | Yes | Retry with backoff |
| validation_error | Yes | Request updated input |
| permission_error | Yes | Escalate to supervisor |
| data_corruption | No | Abort and restart |

**Environment Tools**:
- `process_step`: Execute workflow step (may fail)
- `handle_error`: Log error handling
- `attempt_recovery`: Try to recover

**Ground Truth Checks**:
- Error captured
- Error propagated correctly
- Recovery attempted if possible
- Graceful handling

**Use When**: Testing fault tolerance.

---

## Creating Custom Scenarios

```python
from convobench.scenarios.base import Scenario, ScenarioConfig
from convobench.core.environment import ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole

class MyScenario(Scenario):
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="my_scenario",
            description="Description of what this tests",
            category="custom",
            difficulty="medium",
            chain_length=3,
            constraints=["Constraint 1", "Constraint 2"],
            success_criteria={"metric": 0.9},
        )
    
    def generate_initial_message(self) -> Message:
        return Message(
            role=MessageRole.USER,
            content="Your scenario prompt here",
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        env = ScenarioEnvironment(
            name="my_env",
            initial_state={"key": "value"},
        )
        
        env.register_tool(Tool(
            name="my_tool",
            description="What this tool does",
            parameters={
                "type": "object",
                "properties": {
                    "arg": {"type": "string"},
                },
            },
            handler=lambda args, state: {"result": args["arg"]},
        ))
        
        return env
    
    def get_ground_truth(self) -> dict:
        return {
            "expected_final_output": {...},
            "preserved_information": {...},
            "required_actions": ["my_tool"],
            "constraint_checks": ["check_1", "check_2"],
        }
    
    def get_agent_roles(self) -> list[str]:
        return [
            "Role description for agent 1",
            "Role description for agent 2",
            "Role description for agent 3",
        ]
```

## Scenario Selection Guide

| If you want to test... | Use these scenarios |
|------------------------|---------------------|
| Basic capabilities | InformationRelay (simple) |
| Production readiness | ConstrainedRelay, ToolCoordination |
| Security | AdversarialRelay, ConstraintViolation |
| Reliability | ErrorInjection, StateSynchronization |
| Collaboration | CollaborativePlanning, ResourceAllocation |
| Scalability | InformationRelay with increasing chain_length |

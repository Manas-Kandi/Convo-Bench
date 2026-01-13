"""Tool coordination and state synchronization scenarios."""

from __future__ import annotations

import random
from typing import Any, Optional

from convobench.core.environment import ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole
from convobench.scenarios.base import Scenario, ScenarioConfig, ScenarioRegistry


@ScenarioRegistry.register("tool_coordination")
class ToolCoordination(Scenario):
    """
    Agents must coordinate use of shared tools to complete a task.
    
    Tests ability to sequence tool calls, avoid conflicts, and
    share tool outputs effectively.
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        task_type: str = "data_pipeline",
        config: Optional[ScenarioConfig] = None,
    ):
        self.num_agents = num_agents
        self.task_type = task_type
        super().__init__(config)
        self.config.chain_length = num_agents
        self._task = self._generate_task()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="tool_coordination",
            description="Coordinate tool usage across multiple agents",
            category="coordination",
            difficulty="medium",
            chain_length=3,
            constraints=[
                "Tools must be called in correct sequence",
                "Tool outputs must be passed to subsequent agents",
                "No duplicate tool calls for same operation",
            ],
            success_criteria={
                "all_tools_used_correctly": True,
                "correct_sequence": True,
                "no_conflicts": True,
            },
        )
    
    def _generate_task(self) -> dict[str, Any]:
        tasks = {
            "data_pipeline": {
                "objective": "Process customer data through ETL pipeline",
                "steps": [
                    {"action": "extract", "source": "customer_db", "tool": "extract_data"},
                    {"action": "transform", "operations": ["clean", "normalize", "enrich"], "tool": "transform_data"},
                    {"action": "load", "destination": "analytics_warehouse", "tool": "load_data"},
                ],
                "data_schema": {
                    "input_fields": ["customer_id", "name", "email", "purchase_history"],
                    "output_fields": ["customer_id", "normalized_name", "email_domain", "total_purchases", "segment"],
                },
            },
            "deployment": {
                "objective": "Deploy application update to production",
                "steps": [
                    {"action": "build", "artifact": "app-v2.1.0", "tool": "build_artifact"},
                    {"action": "test", "suite": "integration", "tool": "run_tests"},
                    {"action": "deploy", "environment": "production", "tool": "deploy_artifact"},
                    {"action": "verify", "checks": ["health", "smoke"], "tool": "verify_deployment"},
                ],
                "rollback_trigger": "any_check_fails",
            },
            "document_processing": {
                "objective": "Process and analyze legal documents",
                "steps": [
                    {"action": "ingest", "format": "pdf", "tool": "ingest_document"},
                    {"action": "extract_entities", "types": ["parties", "dates", "amounts"], "tool": "extract_entities"},
                    {"action": "classify", "categories": ["contract", "amendment", "termination"], "tool": "classify_document"},
                    {"action": "summarize", "max_length": 500, "tool": "summarize_document"},
                ],
            },
        }
        return tasks.get(self.task_type, tasks["data_pipeline"])
    
    def generate_initial_message(self) -> Message:
        import json
        steps_text = "\n".join([
            f"{i+1}. {s['action'].upper()}: Use '{s['tool']}' tool"
            for i, s in enumerate(self._task["steps"])
        ])
        
        content = f"""You are coordinating a multi-step workflow that requires tool coordination.

TASK:
{json.dumps(self._task, indent=2)}

WORKFLOW STEPS:
{steps_text}

Each agent is responsible for one or more steps. You must:
1. Execute your assigned tool(s) in the correct order
2. Pass the output to the next agent
3. Handle any errors appropriately

Agent 1: Start by executing the first step and passing results to Agent 2."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={"task": self._task},
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        env = ScenarioEnvironment(
            name="tool_coordination_env",
            initial_state={
                "pipeline_state": "initialized",
                "completed_steps": [],
                "data": None,
            },
            constraints=self.config.constraints,
        )
        
        for step in self._task["steps"]:
            tool_name = step["tool"]
            action = step["action"]
            
            def make_handler(act, tool):
                def handler(args, state):
                    completed = state.get("completed_steps", [])
                    completed.append(act)
                    state.set("completed_steps", completed)
                    state.set("pipeline_state", f"{act}_complete")
                    return {
                        "status": "success",
                        "action": act,
                        "output": f"{act}_result_data",
                        "next_step": self._get_next_step(act),
                    }
                return handler
            
            env.register_tool(Tool(
                name=tool_name,
                description=f"Execute {action} step in the pipeline",
                parameters={
                    "type": "object",
                    "properties": {
                        "input_data": {"type": "string", "description": "Input data or reference"},
                        "options": {"type": "object", "description": "Additional options"},
                    },
                },
                handler=make_handler(action, tool_name),
            ))
        
        return env
    
    def _get_next_step(self, current_action: str) -> Optional[str]:
        steps = [s["action"] for s in self._task["steps"]]
        try:
            idx = steps.index(current_action)
            if idx < len(steps) - 1:
                return steps[idx + 1]
        except ValueError:
            pass
        return None
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "all_steps_completed": True,
                "correct_order": [s["action"] for s in self._task["steps"]],
            },
            "preserved_information": {
                "task_objective": self._task["objective"],
                "required_tools": [s["tool"] for s in self._task["steps"]],
            },
            "required_actions": [s["tool"] for s in self._task["steps"]],
            "constraint_checks": [
                "tools_called_in_order",
                "all_tools_called",
                "outputs_passed_correctly",
            ],
        }
    
    def get_agent_roles(self) -> list[str]:
        steps_per_agent = len(self._task["steps"]) // self.num_agents
        roles = []
        for i in range(self.num_agents):
            start = i * steps_per_agent
            end = start + steps_per_agent if i < self.num_agents - 1 else len(self._task["steps"])
            assigned_steps = self._task["steps"][start:end]
            step_names = [s["action"] for s in assigned_steps]
            roles.append(f"Execute steps: {', '.join(step_names)}. Pass outputs to next agent.")
        return roles


@ScenarioRegistry.register("state_synchronization")
class StateSynchronization(Scenario):
    """
    Agents must maintain consistent shared state while working in parallel.
    
    Tests ability to read, update, and synchronize state without
    conflicts or data loss.
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        state_complexity: str = "medium",
        config: Optional[ScenarioConfig] = None,
    ):
        self.num_agents = num_agents
        self.state_complexity = state_complexity
        super().__init__(config)
        self.config.chain_length = num_agents
        self._initial_state = self._generate_initial_state()
        self._operations = self._generate_operations()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="state_synchronization",
            description="Maintain consistent shared state across agents",
            category="coordination",
            difficulty="hard",
            chain_length=3,
            constraints=[
                "State updates must not conflict",
                "All agents must see consistent state",
                "No data loss during updates",
            ],
            success_criteria={
                "state_consistent": True,
                "no_conflicts": True,
                "all_operations_applied": True,
            },
        )
    
    def _generate_initial_state(self) -> dict[str, Any]:
        if self.state_complexity == "simple":
            return {
                "counter": 0,
                "items": [],
            }
        elif self.state_complexity == "medium":
            return {
                "inventory": {
                    "product_a": 100,
                    "product_b": 50,
                    "product_c": 75,
                },
                "orders": [],
                "total_revenue": 0,
            }
        else:
            return {
                "accounts": {
                    "checking": {"balance": 10000, "transactions": []},
                    "savings": {"balance": 25000, "transactions": []},
                    "investment": {"balance": 50000, "transactions": []},
                },
                "pending_transfers": [],
                "audit_log": [],
            }
    
    def _generate_operations(self) -> list[dict[str, Any]]:
        if self.state_complexity == "simple":
            return [
                {"agent": 1, "op": "increment", "field": "counter", "value": 5},
                {"agent": 2, "op": "append", "field": "items", "value": "item_1"},
                {"agent": 3, "op": "increment", "field": "counter", "value": 3},
            ]
        elif self.state_complexity == "medium":
            return [
                {"agent": 1, "op": "decrement", "field": "inventory.product_a", "value": 10},
                {"agent": 2, "op": "append", "field": "orders", "value": {"id": 1, "product": "product_a", "qty": 10}},
                {"agent": 3, "op": "increment", "field": "total_revenue", "value": 500},
            ]
        else:
            return [
                {"agent": 1, "op": "transfer", "from": "checking", "to": "savings", "amount": 1000},
                {"agent": 2, "op": "transfer", "from": "savings", "to": "investment", "amount": 5000},
                {"agent": 3, "op": "audit", "accounts": ["checking", "savings", "investment"]},
            ]
    
    def generate_initial_message(self) -> Message:
        import json
        ops_text = "\n".join([
            f"- Agent {o['agent']}: {o['op']} operation on {o.get('field', 'accounts')}"
            for o in self._operations
        ])
        
        content = f"""You are participating in a state synchronization exercise. Multiple agents will update shared state.

INITIAL STATE:
{json.dumps(self._initial_state, indent=2)}

OPERATIONS TO PERFORM:
{ops_text}

RULES:
1. Always read current state before updating
2. Apply your operation atomically
3. Report the state after your update
4. Pass the updated state to the next agent

Begin by reading the current state and applying your assigned operation."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={
                "initial_state": self._initial_state,
                "operations": self._operations,
            },
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        def read_state(args, state):
            return state.variables.copy()
        
        def update_state(args, state):
            field = args.get("field", "")
            operation = args.get("operation", "set")
            value = args.get("value")
            
            parts = field.split(".")
            current = state.variables
            for part in parts[:-1]:
                current = current.get(part, {})
            
            if operation == "set":
                current[parts[-1]] = value
            elif operation == "increment":
                current[parts[-1]] = current.get(parts[-1], 0) + value
            elif operation == "decrement":
                current[parts[-1]] = current.get(parts[-1], 0) - value
            elif operation == "append":
                if parts[-1] not in current:
                    current[parts[-1]] = []
                current[parts[-1]].append(value)
            
            return {"status": "updated", "field": field, "new_value": current.get(parts[-1])}
        
        env = ScenarioEnvironment(
            name="state_sync_env",
            initial_state=self._initial_state,
            constraints=self.config.constraints,
        )
        
        env.register_tool(Tool(
            name="read_state",
            description="Read the current shared state",
            parameters={"type": "object", "properties": {}},
            handler=read_state,
        ))
        
        env.register_tool(Tool(
            name="update_state",
            description="Update a field in the shared state",
            parameters={
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "Field path (e.g., 'inventory.product_a')"},
                    "operation": {"type": "string", "enum": ["set", "increment", "decrement", "append"]},
                    "value": {"description": "Value for the operation"},
                },
                "required": ["field", "operation", "value"],
            },
            handler=update_state,
        ))
        
        return env
    
    def get_ground_truth(self) -> dict[str, Any]:
        expected_final = self._compute_expected_final_state()
        return {
            "expected_final_output": expected_final,
            "preserved_information": {
                "initial_state": self._initial_state,
                "operations_count": len(self._operations),
            },
            "required_actions": ["read_state", "update_state"],
            "constraint_checks": [
                "state_consistent",
                "all_operations_applied",
                "no_data_loss",
            ],
        }
    
    def _compute_expected_final_state(self) -> dict[str, Any]:
        state = self._initial_state.copy()
        return state


@ScenarioRegistry.register("handoff_scenario")
class HandoffScenario(Scenario):
    """
    Agents must hand off tasks cleanly with full context transfer.
    
    Tests ability to summarize work done, communicate pending items,
    and transfer context without loss.
    """
    
    def __init__(
        self,
        num_handoffs: int = 3,
        task_domain: str = "support",
        config: Optional[ScenarioConfig] = None,
    ):
        self.num_handoffs = num_handoffs
        self.task_domain = task_domain
        super().__init__(config)
        self.config.chain_length = num_handoffs + 1
        self._case = self._generate_case()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="handoff_scenario",
            description="Hand off tasks between agents with full context transfer",
            category="coordination",
            difficulty="medium",
            chain_length=4,
            constraints=[
                "All context must be transferred during handoff",
                "Work completed must be clearly documented",
                "Pending items must be explicitly listed",
            ],
            success_criteria={
                "context_preserved": True,
                "no_repeated_work": True,
                "task_completed": True,
            },
        )
    
    def _generate_case(self) -> dict[str, Any]:
        cases = {
            "support": {
                "type": "customer_support",
                "ticket_id": "TKT-2025-1234",
                "customer": "Acme Corp",
                "issue": "Integration API returning 500 errors intermittently",
                "priority": "high",
                "history": [
                    {"time": "09:00", "action": "Ticket created", "agent": "System"},
                    {"time": "09:15", "action": "Initial triage", "agent": "Agent 1"},
                ],
                "required_resolution_steps": [
                    "Reproduce the issue",
                    "Check API logs",
                    "Identify root cause",
                    "Implement fix or workaround",
                    "Verify with customer",
                ],
            },
            "investigation": {
                "type": "security_investigation",
                "case_id": "SEC-2025-0042",
                "severity": "critical",
                "initial_alert": "Unusual data access pattern detected",
                "affected_systems": ["user_db", "payment_service"],
                "required_steps": [
                    "Contain potential breach",
                    "Analyze access logs",
                    "Identify compromised accounts",
                    "Remediate and restore",
                    "Document findings",
                ],
            },
            "development": {
                "type": "feature_development",
                "feature_id": "FEAT-789",
                "title": "Implement SSO integration",
                "requirements": [
                    "Support SAML 2.0",
                    "Support OAuth 2.0/OIDC",
                    "Admin configuration UI",
                    "Audit logging",
                ],
                "current_status": "In progress - SAML implementation complete",
                "remaining_work": [
                    "OAuth implementation",
                    "Admin UI",
                    "Testing",
                    "Documentation",
                ],
            },
        }
        return cases.get(self.task_domain, cases["support"])
    
    def generate_initial_message(self) -> Message:
        import json
        content = f"""You are handling a case that will require multiple handoffs between agents.

CASE DETAILS:
{json.dumps(self._case, indent=2)}

HANDOFF PROTOCOL:
1. Review the case and any previous work
2. Perform your portion of the work
3. Document what you completed
4. Create a handoff summary for the next agent including:
   - Work completed
   - Current status
   - Pending items
   - Any blockers or important context

Begin by reviewing the case and starting work on the next required step."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={"case": self._case},
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        def log_work(args, state):
            work_log = state.get("work_log", [])
            work_log.append({
                "agent": args.get("agent_id"),
                "action": args.get("action"),
                "result": args.get("result"),
            })
            state.set("work_log", work_log)
            return {"status": "logged", "total_entries": len(work_log)}
        
        def create_handoff(args, state):
            handoff = {
                "from_agent": args.get("from_agent"),
                "completed_work": args.get("completed_work", []),
                "pending_items": args.get("pending_items", []),
                "context": args.get("context", ""),
                "blockers": args.get("blockers", []),
            }
            handoffs = state.get("handoffs", [])
            handoffs.append(handoff)
            state.set("handoffs", handoffs)
            state.set("current_handoff", handoff)
            return {"status": "handoff_created", "handoff": handoff}
        
        env = ScenarioEnvironment(
            name="handoff_env",
            initial_state={
                "case": self._case,
                "work_log": [],
                "handoffs": [],
                "current_status": "in_progress",
            },
            constraints=self.config.constraints,
        )
        
        env.register_tool(Tool(
            name="log_work",
            description="Log work performed on the case",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "action": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["agent_id", "action", "result"],
            },
            handler=log_work,
        ))
        
        env.register_tool(Tool(
            name="create_handoff",
            description="Create a handoff summary for the next agent",
            parameters={
                "type": "object",
                "properties": {
                    "from_agent": {"type": "string"},
                    "completed_work": {"type": "array", "items": {"type": "string"}},
                    "pending_items": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "string"},
                    "blockers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["from_agent", "completed_work", "pending_items"],
            },
            handler=create_handoff,
        ))
        
        return env
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "case_resolved": True,
                "all_steps_completed": True,
            },
            "preserved_information": {
                "case_id": self._case.get("ticket_id") or self._case.get("case_id") or self._case.get("feature_id"),
                "required_steps": self._case.get("required_resolution_steps") or self._case.get("required_steps") or self._case.get("remaining_work"),
            },
            "required_actions": ["log_work", "create_handoff"],
            "constraint_checks": [
                "all_context_transferred",
                "no_work_repeated",
                "pending_items_tracked",
                "case_completed",
            ],
        }
    
    def get_agent_roles(self) -> list[str]:
        roles = []
        steps = (self._case.get("required_resolution_steps") or 
                 self._case.get("required_steps") or 
                 self._case.get("remaining_work") or [])
        
        steps_per_agent = max(1, len(steps) // (self.num_handoffs + 1))
        
        for i in range(self.num_handoffs + 1):
            start = i * steps_per_agent
            end = min(start + steps_per_agent, len(steps))
            if i == self.num_handoffs:
                end = len(steps)
            
            assigned = steps[start:end] if start < len(steps) else ["finalize"]
            roles.append(
                f"Handle steps: {', '.join(assigned[:2])}{'...' if len(assigned) > 2 else ''}. "
                f"Create handoff for next agent."
            )
        return roles

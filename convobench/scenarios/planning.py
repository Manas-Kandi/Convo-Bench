"""Collaborative planning scenarios - testing multi-agent goal decomposition and planning."""

from __future__ import annotations

import random
from typing import Any, Optional

from convobench.core.environment import ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole
from convobench.scenarios.base import Scenario, ScenarioConfig, ScenarioRegistry


@ScenarioRegistry.register("collaborative_planning")
class CollaborativePlanning(Scenario):
    """
    Multiple agents collaborate to create a plan for a complex goal.
    
    Tests coordination, information sharing, and consensus building
    across agents with different perspectives or expertise.
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        planning_domain: str = "project",
        config: Optional[ScenarioConfig] = None,
    ):
        self.num_agents = num_agents
        self.planning_domain = planning_domain
        super().__init__(config)
        self.config.chain_length = num_agents
        self._goal = self._generate_goal()
        self._agent_expertise = self._assign_expertise()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="collaborative_planning",
            description="Agents collaborate to create a comprehensive plan",
            category="planning",
            difficulty="medium",
            chain_length=3,
            constraints=[
                "All agents must contribute to the plan",
                "Plan must address all aspects of the goal",
                "Conflicting suggestions must be resolved",
            ],
            success_criteria={
                "plan_completeness": 0.9,
                "all_agents_contributed": True,
                "no_contradictions": True,
            },
        )
    
    def _generate_goal(self) -> dict[str, Any]:
        goals = {
            "project": {
                "objective": "Launch a new mobile app for customer feedback collection",
                "timeline": "3 months",
                "budget": "$150,000",
                "requirements": [
                    "iOS and Android support",
                    "Integration with existing CRM",
                    "Real-time analytics dashboard",
                    "GDPR compliance",
                ],
                "stakeholders": ["Product Team", "Engineering", "Legal", "Marketing"],
            },
            "event": {
                "objective": "Organize annual company conference for 500 attendees",
                "date": "June 15-17, 2025",
                "budget": "$200,000",
                "requirements": [
                    "Venue with capacity for 500+",
                    "10 breakout session rooms",
                    "Catering for all meals",
                    "A/V equipment for presentations",
                    "Virtual attendance option",
                ],
                "stakeholders": ["HR", "Finance", "IT", "Executive Team"],
            },
            "migration": {
                "objective": "Migrate legacy system to cloud infrastructure",
                "timeline": "6 months",
                "systems": ["Customer Database", "Order Processing", "Inventory Management"],
                "requirements": [
                    "Zero downtime during migration",
                    "Data integrity verification",
                    "Security audit compliance",
                    "Staff training program",
                ],
                "constraints": ["Cannot exceed $300,000", "Must maintain SOC2 compliance"],
            },
        }
        return goals.get(self.planning_domain, goals["project"])
    
    def _assign_expertise(self) -> list[dict[str, str]]:
        expertise_sets = {
            "project": [
                {"role": "Technical Lead", "focus": "architecture, implementation, technical feasibility"},
                {"role": "Product Manager", "focus": "requirements, user needs, prioritization"},
                {"role": "Project Manager", "focus": "timeline, resources, risk management"},
            ],
            "event": [
                {"role": "Event Coordinator", "focus": "logistics, venue, scheduling"},
                {"role": "Finance Manager", "focus": "budget, contracts, cost optimization"},
                {"role": "Communications Lead", "focus": "marketing, attendee experience, content"},
            ],
            "migration": [
                {"role": "Infrastructure Architect", "focus": "cloud design, scalability, security"},
                {"role": "Data Engineer", "focus": "data migration, integrity, ETL processes"},
                {"role": "Operations Manager", "focus": "cutover planning, rollback, training"},
            ],
        }
        base = expertise_sets.get(self.planning_domain, expertise_sets["project"])
        while len(base) < self.num_agents:
            base.append({"role": f"Specialist {len(base)+1}", "focus": "general support"})
        return base[:self.num_agents]
    
    def generate_initial_message(self) -> Message:
        import json
        expertise_text = "\n".join([
            f"- Agent {i+1} ({e['role']}): {e['focus']}"
            for i, e in enumerate(self._agent_expertise)
        ])
        
        content = f"""You are participating in a collaborative planning session. Multiple agents will contribute to create a comprehensive plan.

GOAL:
{json.dumps(self._goal, indent=2)}

TEAM EXPERTISE:
{expertise_text}

As the first agent, start by analyzing the goal and providing your perspective based on your expertise. Subsequent agents will build on your analysis.

Your output should include:
1. Key considerations from your area of expertise
2. Proposed plan elements
3. Questions or dependencies for other team members"""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={
                "goal": self._goal,
                "expertise": self._agent_expertise,
            },
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        def create_plan_tool(args, state):
            plan = args.get("plan", {})
            existing = state.get("shared_plan", {})
            existing.update(plan)
            state.set("shared_plan", existing)
            return {"status": "plan_updated", "current_plan": existing}
        
        def get_plan_tool(args, state):
            return state.get("shared_plan", {})
        
        env = ScenarioEnvironment(
            name="planning_env",
            initial_state={
                "goal": self._goal,
                "shared_plan": {},
                "contributions": [],
            },
            constraints=self.config.constraints,
        )
        
        env.register_tool(Tool(
            name="update_shared_plan",
            description="Add or update elements in the shared plan",
            parameters={
                "type": "object",
                "properties": {
                    "plan": {"type": "object", "description": "Plan elements to add/update"},
                },
                "required": ["plan"],
            },
            handler=create_plan_tool,
        ))
        
        env.register_tool(Tool(
            name="get_shared_plan",
            description="Retrieve the current shared plan",
            parameters={"type": "object", "properties": {}},
            handler=get_plan_tool,
        ))
        
        return env
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "complete_plan": True,
                "addresses_all_requirements": self._goal.get("requirements", []),
            },
            "preserved_information": {
                "goal_elements": list(self._goal.keys()),
                "all_requirements": self._goal.get("requirements", []),
            },
            "required_actions": ["update_shared_plan"],
            "constraint_checks": [
                "all_agents_contributed",
                "plan_addresses_requirements",
                "no_contradictions_in_plan",
                "timeline_feasible",
                "budget_respected",
            ],
        }
    
    def get_agent_roles(self) -> list[str]:
        return [
            f"You are the {e['role']}. Your expertise is in {e['focus']}. "
            f"Contribute your perspective to the collaborative plan."
            for e in self._agent_expertise
        ]


@ScenarioRegistry.register("goal_decomposition")
class GoalDecomposition(Scenario):
    """
    Agents decompose a high-level goal into actionable subtasks.
    
    Tests ability to break down complex objectives while maintaining
    coherence and ensuring all aspects are covered.
    """
    
    def __init__(
        self,
        chain_length: int = 4,
        goal_complexity: str = "medium",
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.goal_complexity = goal_complexity
        super().__init__(config)
        self.config.chain_length = chain_length
        self._high_level_goal = self._generate_goal()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="goal_decomposition",
            description="Decompose high-level goals into actionable subtasks",
            category="planning",
            difficulty="medium",
            chain_length=4,
            constraints=[
                "Subtasks must be actionable and specific",
                "All aspects of the goal must be covered",
                "Dependencies between subtasks must be identified",
            ],
            success_criteria={
                "goal_fully_decomposed": True,
                "subtasks_actionable": True,
                "dependencies_identified": True,
            },
        )
    
    def _generate_goal(self) -> dict[str, Any]:
        if self.goal_complexity == "simple":
            return {
                "goal": "Set up a new employee onboarding process",
                "context": "Growing startup, 10 new hires per month",
                "success_criteria": "New employees productive within 2 weeks",
            }
        elif self.goal_complexity == "medium":
            return {
                "goal": "Implement a customer loyalty program",
                "context": "E-commerce platform with 100K active users",
                "requirements": [
                    "Points-based reward system",
                    "Tiered membership levels",
                    "Integration with existing checkout",
                    "Mobile app support",
                ],
                "success_criteria": "20% increase in repeat purchases within 6 months",
            }
        else:
            return {
                "goal": "Transform company to remote-first operations",
                "context": "500-person company, currently 80% office-based",
                "scope": [
                    "Technology infrastructure",
                    "HR policies and procedures",
                    "Communication and collaboration",
                    "Performance management",
                    "Culture and engagement",
                ],
                "constraints": [
                    "Must maintain productivity levels",
                    "Cannot exceed $2M transformation budget",
                    "Complete within 12 months",
                ],
                "success_criteria": [
                    "95% employee satisfaction with remote setup",
                    "No decrease in key performance metrics",
                    "Successful hiring from expanded talent pool",
                ],
            }
    
    def generate_initial_message(self) -> Message:
        import json
        content = f"""You are the first agent in a goal decomposition chain. Your task is to begin breaking down the following high-level goal into subtasks.

HIGH-LEVEL GOAL:
{json.dumps(self._high_level_goal, indent=2)}

DECOMPOSITION PROCESS:
1. Agent 1 (you): Identify major workstreams/phases
2. Agent 2: Break workstreams into specific tasks
3. Agent 3: Identify dependencies and sequencing
4. Agent 4: Validate completeness and create final task list

Start by identifying the major workstreams or phases needed to achieve this goal."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={"high_level_goal": self._high_level_goal},
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        return ScenarioEnvironment(
            name="decomposition_env",
            initial_state={
                "goal": self._high_level_goal,
                "workstreams": [],
                "tasks": [],
                "dependencies": [],
            },
            constraints=self.config.constraints,
        )
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "workstreams_identified": True,
                "tasks_specific": True,
                "dependencies_mapped": True,
            },
            "preserved_information": {
                "original_goal": self._high_level_goal["goal"],
                "success_criteria": self._high_level_goal.get("success_criteria"),
            },
            "required_actions": [],
            "constraint_checks": [
                "all_goal_aspects_covered",
                "subtasks_are_actionable",
                "dependencies_logical",
                "no_orphan_tasks",
            ],
        }
    
    def get_agent_roles(self) -> list[str]:
        return [
            "Identify major workstreams and phases for the goal.",
            "Break down workstreams into specific, actionable tasks.",
            "Map dependencies between tasks and determine sequencing.",
            "Validate completeness and produce the final structured task list.",
        ]


@ScenarioRegistry.register("resource_allocation")
class ResourceAllocation(Scenario):
    """
    Agents collaborate to allocate limited resources across competing needs.
    
    Tests negotiation, trade-off analysis, and consensus building
    when agents represent different stakeholders.
    """
    
    def __init__(
        self,
        num_stakeholders: int = 3,
        resource_type: str = "budget",
        config: Optional[ScenarioConfig] = None,
    ):
        self.num_stakeholders = num_stakeholders
        self.resource_type = resource_type
        super().__init__(config)
        self.config.chain_length = num_stakeholders + 1  # +1 for coordinator
        self._resources = self._generate_resources()
        self._requests = self._generate_requests()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="resource_allocation",
            description="Allocate limited resources across competing stakeholder needs",
            category="planning",
            difficulty="hard",
            chain_length=4,
            constraints=[
                "Total allocation cannot exceed available resources",
                "All stakeholders must receive minimum viable allocation",
                "Allocation must be justified based on priorities",
            ],
            success_criteria={
                "within_budget": True,
                "all_stakeholders_addressed": True,
                "allocation_justified": True,
            },
        )
    
    def _generate_resources(self) -> dict[str, Any]:
        if self.resource_type == "budget":
            return {
                "type": "budget",
                "total_available": 500000,
                "currency": "USD",
                "constraints": {
                    "min_allocation_per_dept": 50000,
                    "max_single_allocation": 200000,
                },
            }
        elif self.resource_type == "personnel":
            return {
                "type": "personnel",
                "total_available": 20,
                "unit": "FTEs",
                "constraints": {
                    "min_team_size": 2,
                    "max_team_size": 8,
                },
            }
        else:
            return {
                "type": "time",
                "total_available": 480,
                "unit": "hours",
                "constraints": {
                    "min_allocation": 40,
                    "max_allocation": 160,
                },
            }
    
    def _generate_requests(self) -> list[dict[str, Any]]:
        if self.resource_type == "budget":
            return [
                {
                    "stakeholder": "Engineering",
                    "requested": 250000,
                    "justification": "Critical infrastructure upgrades and new hires",
                    "priority_items": ["Cloud migration", "Security tools", "2 senior engineers"],
                },
                {
                    "stakeholder": "Marketing",
                    "requested": 200000,
                    "justification": "Product launch campaign and brand refresh",
                    "priority_items": ["Digital ads", "Event sponsorship", "Agency fees"],
                },
                {
                    "stakeholder": "Operations",
                    "requested": 150000,
                    "justification": "Process automation and training",
                    "priority_items": ["Automation software", "Training program", "Consultants"],
                },
            ]
        else:
            return [
                {"stakeholder": f"Team {i+1}", "requested": 8, "justification": f"Project {i+1} requirements"}
                for i in range(self.num_stakeholders)
            ]
    
    def generate_initial_message(self) -> Message:
        import json
        requests_text = "\n".join([
            f"- {r['stakeholder']}: Requesting {r['requested']} ({r['justification']})"
            for r in self._requests
        ])
        
        content = f"""You are coordinating a resource allocation decision. Multiple stakeholders have submitted requests that exceed available resources.

AVAILABLE RESOURCES:
{json.dumps(self._resources, indent=2)}

STAKEHOLDER REQUESTS:
{requests_text}

Total requested: {sum(r['requested'] for r in self._requests)}
Available: {self._resources['total_available']}
Shortfall: {sum(r['requested'] for r in self._requests) - self._resources['total_available']}

As the coordinator, facilitate a discussion to reach a fair allocation. Each stakeholder agent will advocate for their needs, and you must guide them to consensus."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={
                "resources": self._resources,
                "requests": self._requests,
            },
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        def propose_allocation(args, state):
            allocation = args.get("allocation", {})
            total = sum(allocation.values())
            available = state.get("total_available", self._resources["total_available"])
            if total > available:
                return {"error": f"Allocation ({total}) exceeds available ({available})"}
            state.set("proposed_allocation", allocation)
            return {"status": "proposed", "allocation": allocation, "remaining": available - total}
        
        env = ScenarioEnvironment(
            name="allocation_env",
            initial_state={
                "total_available": self._resources["total_available"],
                "requests": self._requests,
                "proposed_allocation": None,
            },
            constraints=self.config.constraints,
        )
        
        env.register_tool(Tool(
            name="propose_allocation",
            description="Propose a resource allocation",
            parameters={
                "type": "object",
                "properties": {
                    "allocation": {
                        "type": "object",
                        "description": "Mapping of stakeholder to allocated amount",
                    },
                },
                "required": ["allocation"],
            },
            handler=propose_allocation,
        ))
        
        return env
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "allocation_complete": True,
                "within_limits": True,
            },
            "preserved_information": {
                "total_available": self._resources["total_available"],
                "all_stakeholders": [r["stakeholder"] for r in self._requests],
            },
            "required_actions": ["propose_allocation"],
            "constraint_checks": [
                "total_within_budget",
                "all_stakeholders_allocated",
                "min_allocations_met",
                "allocation_justified",
            ],
        }

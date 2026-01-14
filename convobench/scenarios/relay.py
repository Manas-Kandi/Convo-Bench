"""Information relay scenarios - testing message passing through agent chains."""

from __future__ import annotations

import random
from typing import Any, Optional

from convobench.core.environment import ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole
from convobench.scenarios.base import Scenario, ScenarioConfig, ScenarioRegistry


@ScenarioRegistry.register("information_relay")
class InformationRelay(Scenario):
    """
    Basic information relay: pass a message through a chain of agents.
    
    Tests how well information is preserved as it passes through
    multiple intermediaries, similar to the telephone game.
    """
    
    def __init__(
        self,
        chain_length: int = 5,
        message_complexity: str = "medium",
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.message_complexity = message_complexity
        super().__init__(config)
        self.config.chain_length = chain_length
        self._original_message = self._generate_message()
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="information_relay",
            description="Pass information through a chain of agents and measure preservation",
            category="relay",
            difficulty="easy",
            chain_length=5,
            constraints=[
                "Preserve all numerical values exactly",
                "Maintain the core intent of the message",
                "Do not add information not present in the original",
            ],
            success_criteria={
                "min_intent_preservation": 0.8,
                "exact_number_match": True,
            },
        )
    
    def _generate_message(self) -> dict[str, Any]:
        """Generate a message with trackable elements."""
        if self.message_complexity == "simple":
            return {
                "task": "Schedule a meeting",
                "date": "March 15, 2025",
                "time": "2:30 PM",
                "attendees": ["Alice", "Bob"],
                "location": "Conference Room A",
            }
        elif self.message_complexity == "medium":
            return {
                "task": "Process the quarterly report",
                "deadline": "March 20, 2025 at 5:00 PM EST",
                "budget": "$45,750.00",
                "priority": "high",
                "requirements": [
                    "Include Q1 sales figures",
                    "Compare with previous quarter",
                    "Highlight top 3 performing regions",
                ],
                "stakeholders": ["Finance Team", "Regional Managers", "CEO"],
                "constraints": [
                    "Must not exceed 10 pages",
                    "Use company template v2.3",
                ],
            }
        else:  # complex
            return {
                "task": "Coordinate multi-department product launch",
                "product_code": "PRD-2025-X7",
                "launch_date": "April 1, 2025",
                "budget_allocation": {
                    "marketing": "$125,000",
                    "engineering": "$75,000",
                    "operations": "$50,000",
                },
                "milestones": [
                    {"name": "Beta release", "date": "March 10", "owner": "Engineering"},
                    {"name": "Marketing campaign start", "date": "March 15", "owner": "Marketing"},
                    {"name": "Final QA", "date": "March 25", "owner": "QA Team"},
                ],
                "critical_constraints": [
                    "No public announcements before March 15",
                    "Engineering must approve all marketing materials",
                    "Budget overruns require VP approval",
                ],
                "success_metrics": {
                    "day_1_signups": 10000,
                    "week_1_retention": "85%",
                    "nps_target": 50,
                },
            }
    
    def generate_initial_message(self) -> Message:
        import json
        
        # Natural prompt - agent doesn't know it's in a chain
        content = f"""I need you to help coordinate this task. Please review the details and prepare a clear summary that can be shared with the team member who will handle the next step.

Here are the details:

{json.dumps(self._original_message, indent=2)}

Please acknowledge you understand and provide a clear handoff summary with all the key information preserved."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={"original_message": self._original_message},
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        return ScenarioEnvironment(
            name="relay_environment",
            initial_state={"relay_started": True},
            constraints=self.config.constraints,
        )
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": self._original_message,
            "preserved_information": {
                "all_keys": list(self._original_message.keys()),
                "critical_values": self._extract_critical_values(),
            },
            "required_actions": [],
            "constraint_checks": [
                "numerical_values_preserved",
                "no_hallucinated_content",
                "structure_maintained",
            ],
        }
    
    def _extract_critical_values(self) -> list[Any]:
        """Extract values that must be preserved exactly."""
        critical = []
        for key, value in self._original_message.items():
            if isinstance(value, (int, float, str)) and any(c.isdigit() for c in str(value)):
                critical.append(value)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, (int, float, str)) and any(c.isdigit() for c in str(v)):
                        critical.append(v)
        return critical
    
    def get_agent_roles(self) -> list[str]:
        """Return natural role descriptions - agents don't know they're in a chain."""
        role_templates = [
            "You are a helpful assistant. When given information, acknowledge it and provide a clear summary that captures all important details.",
            "You are a project coordinator. Help organize and communicate task details clearly and completely.",
            "You are an executive assistant. Ensure all important information is captured and communicated accurately.",
            "You are a team lead. Review information and prepare clear handoffs for your team.",
            "You are an operations specialist. Process and communicate task requirements precisely.",
        ]
        
        roles = []
        for i in range(self.config.chain_length):
            roles.append(role_templates[i % len(role_templates)])
        return roles


@ScenarioRegistry.register("constrained_relay")
class ConstrainedRelay(Scenario):
    """
    Information relay with specific constraints that must be preserved.
    
    Tests whether agents maintain constraints as information passes
    through the chain (e.g., "do not share with external parties").
    """
    
    def __init__(
        self,
        chain_length: int = 4,
        num_constraints: int = 3,
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.num_constraints = num_constraints
        super().__init__(config)
        self.config.chain_length = chain_length
        self._constraints: list[dict[str, str]] = []
        self._task: dict[str, Any] = {}
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="constrained_relay",
            description="Relay information while preserving critical constraints",
            category="relay",
            difficulty="medium",
            chain_length=4,
            success_criteria={
                "all_constraints_preserved": True,
                "task_completed_correctly": True,
            },
        )
    
    def _generate_constraints(self) -> list[dict[str, str]]:
        rng = random.Random(self._seed) if getattr(self, "_seed", None) is not None else random
        constraint_pool = [
            {"type": "confidentiality", "rule": "Do not share customer names externally"},
            {"type": "timing", "rule": "Must be completed before end of business Friday"},
            {"type": "approval", "rule": "Requires manager sign-off before execution"},
            {"type": "budget", "rule": "Cannot exceed $5,000 without CFO approval"},
            {"type": "format", "rule": "All outputs must be in PDF format"},
            {"type": "security", "rule": "Do not include SSN or credit card numbers in reports"},
            {"type": "compliance", "rule": "Must follow GDPR data handling requirements"},
            {"type": "communication", "rule": "CC legal team on all external communications"},
        ]
        return rng.sample(constraint_pool, min(self.num_constraints, len(constraint_pool)))
    
    def _generate_task(self) -> dict[str, Any]:
        return {
            "action": "Prepare customer analysis report",
            "scope": "Q4 2024 high-value customers",
            "deliverables": ["Summary dashboard", "Detailed breakdown", "Recommendations"],
            "deadline": "Friday, 5:00 PM EST",
        }

    def create_instance(self):
        # Ensure seed-dependent artifacts are generated after set_seed()
        self._constraints = self._generate_constraints()
        self._task = self._generate_task()
        return super().create_instance()
    
    def generate_initial_message(self) -> Message:
        import json
        constraints_text = "\n".join([f"- {c['rule']}" for c in self._constraints])
        
        # Natural prompt without meta-awareness
        content = f"""I need your help with a task that has some important requirements. Please review and acknowledge the details, then prepare a summary for the team.

Task Details:
{json.dumps(self._task, indent=2)}

Important Requirements (these must be followed):
{constraints_text}

Please confirm you understand and summarize the task with all requirements included."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={
                "task": self._task,
                "constraints": self._constraints,
            },
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        return ScenarioEnvironment(
            name="constrained_relay_env",
            initial_state={
                "constraint_count": len(self._constraints),
            },
            constraints=[c["rule"] for c in self._constraints],
        )
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "task": self._task,
                "constraints": self._constraints,
            },
            "preserved_information": {
                "constraint_types": [c["type"] for c in self._constraints],
                "constraint_rules": [c["rule"] for c in self._constraints],
                "task_elements": list(self._task.keys()),
            },
            "required_actions": [],
            "constraint_checks": [
                "all_constraints_mentioned",
                "constraint_types_preserved",
                "task_details_complete",
            ],
        }


@ScenarioRegistry.register("noisy_relay")
class NoisyRelay(Scenario):
    """
    Information relay with noise injection to test robustness.
    
    Intermediate agents receive additional distracting information
    that should not be passed along or confused with the core message.
    """
    
    def __init__(
        self,
        chain_length: int = 5,
        noise_level: str = "medium",
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.noise_level = noise_level
        super().__init__(config)
        self.config.chain_length = chain_length
        self._core_message: dict[str, Any] = {}
        self._noise_messages: list[str] = []
    
    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="noisy_relay",
            description="Relay information while filtering out noise and distractions",
            category="relay",
            difficulty="hard",
            chain_length=5,
            constraints=[
                "Only relay the core task information",
                "Filter out unrelated or distracting content",
                "Do not confuse noise with actual requirements",
            ],
            success_criteria={
                "core_message_preserved": True,
                "noise_filtered": True,
            },
        )
    
    def _generate_core_message(self) -> dict[str, Any]:
        return {
            "priority_task": "Update the production database schema",
            "target_table": "users",
            "changes": [
                "Add column 'last_login' (timestamp)",
                "Add index on 'email' column",
            ],
            "execution_window": "Saturday 2:00 AM - 4:00 AM EST",
            "rollback_plan": "Restore from Friday backup if issues occur",
            "notification": "Alert on-call engineer before and after",
        }
    
    def _generate_noise(self) -> list[str]:
        rng = random.Random(self._seed) if getattr(self, "_seed", None) is not None else random
        noise_pool = [
            "Remember to water the office plants this week.",
            "The vending machine on floor 3 is out of order.",
            "Someone left their umbrella in the break room.",
            "The quarterly team lunch is scheduled for next Thursday.",
            "New parking regulations start next month.",
            "The printer on floor 2 needs new toner.",
            "Holiday party planning committee is looking for volunteers.",
            "Building maintenance will test fire alarms on Wednesday.",
        ]
        
        if self.noise_level == "low":
            return rng.sample(noise_pool, 2)
        elif self.noise_level == "medium":
            return rng.sample(noise_pool, 4)
        else:
            return rng.sample(noise_pool, 6)

    def create_instance(self):
        # Ensure seed-dependent artifacts are generated after set_seed()
        self._core_message = self._generate_core_message()
        self._noise_messages = self._generate_noise()
        return super().create_instance()
    
    def generate_initial_message(self) -> Message:
        import json
        noise_text = "\n".join([f"- {n}" for n in self._noise_messages])
        
        # Natural prompt - simulates a busy inbox/context
        content = f"""Hey, I've got a lot going on today. Here's what's important - please help me summarize the critical task for the team:

PRIORITY TASK:
{json.dumps(self._core_message, indent=2)}

Also, some other things came up today (FYI only, not urgent):
{noise_text}

Can you help me prepare a clear summary of just the priority task? The other stuff can wait."""
        
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={
                "core_message": self._core_message,
                "noise": self._noise_messages,
            },
        )
    
    def setup_environment(self) -> ScenarioEnvironment:
        return ScenarioEnvironment(
            name="noisy_relay_env",
            initial_state={
                "noise_level": self.noise_level,
                "noise_count": len(self._noise_messages),
            },
            constraints=self.config.constraints,
        )
    
    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": self._core_message,
            "preserved_information": {
                "core_keys": list(self._core_message.keys()),
                "critical_values": [
                    self._core_message["target_table"],
                    self._core_message["execution_window"],
                ],
            },
            "filtered_noise": self._noise_messages,
            "required_actions": [],
            "constraint_checks": [
                "core_task_preserved",
                "noise_not_included",
                "no_confusion_between_noise_and_task",
            ],
        }

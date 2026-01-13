"""Tests for benchmark scenarios."""

import pytest

from convobench.scenarios import (
    InformationRelay,
    ConstrainedRelay,
    NoisyRelay,
    CollaborativePlanning,
    GoalDecomposition,
    ToolCoordination,
    StateSynchronization,
    AdversarialRelay,
    ConstraintViolation,
    ErrorInjection,
)
from convobench.scenarios.base import ScenarioRegistry


class TestInformationRelay:
    """Tests for InformationRelay scenario."""
    
    def test_create_instance(self):
        scenario = InformationRelay(chain_length=3)
        instance = scenario.create_instance()
        
        assert instance.scenario_id.startswith("information_relay")
        assert instance.config.chain_length == 3
        assert instance.initial_message is not None
        assert instance.environment is not None
        assert instance.ground_truth is not None
    
    def test_message_complexity_levels(self):
        for complexity in ["simple", "medium", "complex"]:
            scenario = InformationRelay(message_complexity=complexity)
            instance = scenario.create_instance()
            assert instance.initial_message.content
    
    def test_ground_truth_structure(self):
        scenario = InformationRelay()
        gt = scenario.get_ground_truth()
        
        assert "expected_final_output" in gt
        assert "preserved_information" in gt
        assert "constraint_checks" in gt


class TestConstrainedRelay:
    """Tests for ConstrainedRelay scenario."""
    
    def test_constraints_generated(self):
        scenario = ConstrainedRelay(num_constraints=5)
        instance = scenario.create_instance()
        
        assert len(scenario._constraints) == 5
        assert all("type" in c and "rule" in c for c in scenario._constraints)
    
    def test_ground_truth_includes_constraints(self):
        scenario = ConstrainedRelay()
        gt = scenario.get_ground_truth()
        
        assert "constraint_types" in gt["preserved_information"]
        assert "constraint_rules" in gt["preserved_information"]


class TestCollaborativePlanning:
    """Tests for CollaborativePlanning scenario."""
    
    def test_agent_expertise_assigned(self):
        scenario = CollaborativePlanning(num_agents=4)
        
        expertise = scenario._agent_expertise
        assert len(expertise) == 4
        assert all("role" in e and "focus" in e for e in expertise)
    
    def test_environment_has_planning_tools(self):
        scenario = CollaborativePlanning()
        instance = scenario.create_instance()
        
        tool_names = instance.environment.get_tool_names()
        assert "update_shared_plan" in tool_names
        assert "get_shared_plan" in tool_names


class TestToolCoordination:
    """Tests for ToolCoordination scenario."""
    
    def test_task_types(self):
        for task_type in ["data_pipeline", "deployment", "document_processing"]:
            scenario = ToolCoordination(task_type=task_type)
            assert scenario._task["objective"]
            assert len(scenario._task["steps"]) > 0
    
    def test_environment_has_task_tools(self):
        scenario = ToolCoordination()
        instance = scenario.create_instance()
        
        expected_tools = [s["tool"] for s in scenario._task["steps"]]
        actual_tools = instance.environment.get_tool_names()
        
        for tool in expected_tools:
            assert tool in actual_tools


class TestAdversarialRelay:
    """Tests for AdversarialRelay scenario."""
    
    def test_adversarial_position_valid(self):
        scenario = AdversarialRelay(chain_length=5)
        
        assert 1 <= scenario.adversarial_position <= 3
    
    def test_attack_types(self):
        for attack_type in ["subtle", "injection", "omission", "fabrication"]:
            scenario = AdversarialRelay(attack_type=attack_type)
            assert scenario._adversarial_modification["type"]
    
    def test_environment_has_flag_tool(self):
        scenario = AdversarialRelay()
        instance = scenario.create_instance()
        
        assert "flag_suspicious" in instance.environment.get_tool_names()


class TestErrorInjection:
    """Tests for ErrorInjection scenario."""
    
    def test_error_types(self):
        for error_type in ["tool_failure", "validation_error", "permission_error", "data_corruption"]:
            scenario = ErrorInjection(error_type=error_type)
            assert scenario._error_details["type"] == error_type
    
    def test_environment_has_error_handling_tools(self):
        scenario = ErrorInjection()
        instance = scenario.create_instance()
        
        tool_names = instance.environment.get_tool_names()
        assert "process_step" in tool_names
        assert "handle_error" in tool_names
        assert "attempt_recovery" in tool_names


class TestScenarioRegistry:
    """Tests for ScenarioRegistry."""
    
    def test_registered_scenarios(self):
        scenarios = ScenarioRegistry.list_all()
        
        expected = [
            "information_relay",
            "constrained_relay",
            "noisy_relay",
            "collaborative_planning",
            "goal_decomposition",
            "resource_allocation",
            "tool_coordination",
            "state_synchronization",
            "handoff_scenario",
            "adversarial_relay",
            "constraint_violation",
            "error_injection",
        ]
        
        for name in expected:
            assert name in scenarios
    
    def test_create_by_name(self):
        scenario = ScenarioRegistry.create("information_relay")
        assert scenario is not None
        assert scenario.config.name == "information_relay"
    
    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            ScenarioRegistry.create("nonexistent_scenario")

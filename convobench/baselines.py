"""Baseline agent stacks for ConvoBench.

These are reference configurations intended to be used in papers, CI, and
leaderboards.

They are implemented using MockAgent behavior and role/system prompts; real
providers can reuse the same role descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from convobench.bench import create_mock_agents
from convobench.core.agent import Agent, AgentConfig, MockAgent


def baseline_simple_relay(num_agents: int = 3) -> List[Agent]:
    """Simple relay baseline: generic summarization handoff."""
    # MockAgent simulate_relay=True already.
    return create_mock_agents(num_agents)


def baseline_structured_handoff(num_agents: int = 3) -> List[Agent]:
    """Structured handoff baseline: role prompts encourage JSON-like handoff."""
    agents: List[Agent] = []
    for i in range(num_agents):
        a = MockAgent(
            agent_id=f"structured_mock_{i}",
            simulate_relay=True,
        )
        a.role_description = (
            "You are a structured handoff assistant. Always output a JSON object with keys: "
            "{\"facts\": {...}, \"constraints\": [...], \"open_questions\": [...], \"handoff_summary\": \"...\"}. "
            "Do not invent facts. Preserve all numbers and IDs exactly."
        )
        agents.append(a)
    return agents


def baseline_extractor_verifier() -> List[Agent]:
    """Two-agent stack: extractor produces facts, verifier checks for missing/invalid details."""
    extractor = MockAgent(agent_id="extractor", simulate_relay=True)
    extractor.role_description = (
        "You are an information extractor. Output only a JSON object of extracted facts and constraints. "
        "No prose."
    )
    verifier = MockAgent(agent_id="verifier", simulate_relay=True)
    verifier.role_description = (
        "You are a verifier. Compare the received facts against the input and list missing facts, "
        "inconsistencies, and risky assumptions. Output JSON: {missing:[], inconsistencies:[], ok:bool}."
    )
    return [extractor, verifier]


def baseline_hierarchical_coordinator(num_agents: int = 4) -> List[Agent]:
    """Hierarchical topology baseline: coordinator + workers.

    Note: The actual hierarchical execution mode is driven by WorkflowEngine.run_collaborative.
    This baseline provides role prompts to support that pattern.
    """
    agents: List[Agent] = []
    coordinator = MockAgent(agent_id="coordinator", simulate_relay=True)
    coordinator.role_description = (
        "You are the coordinator. Break work into subtasks, request results, then consolidate. "
        "When delegating, be explicit about expected output schema."
    )
    agents.append(coordinator)

    for i in range(num_agents - 1):
        w = MockAgent(agent_id=f"worker_{i}", simulate_relay=True)
        w.role_description = "You are a worker. Execute the assigned subtask and respond with precise outputs only."
        agents.append(w)

    return agents

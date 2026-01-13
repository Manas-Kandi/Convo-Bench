"""Tests for workflow engine."""

import pytest

from convobench.core.agent import AgentChain, MockAgent
from convobench.core.engine import WorkflowConfig, WorkflowEngine
from convobench.core.environment import Environment
from convobench.core.metrics import MetricsCollector
from convobench.core.types import Message, MessageRole, WorkflowStatus


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    return [
        MockAgent(agent_id="agent_1", responses=["Response from agent 1"]),
        MockAgent(agent_id="agent_2", responses=["Response from agent 2"]),
        MockAgent(agent_id="agent_3", responses=["Response from agent 3"]),
    ]


@pytest.fixture
def environment():
    """Create test environment."""
    return Environment(name="test_env")


@pytest.fixture
def initial_message():
    """Create initial test message."""
    return Message(
        role=MessageRole.USER,
        content="Test message to relay through agents",
    )


class TestWorkflowEngine:
    """Tests for WorkflowEngine."""
    
    @pytest.mark.asyncio
    async def test_run_chain_basic(self, mock_agents, environment, initial_message):
        engine = WorkflowEngine()
        chain = AgentChain(mock_agents)
        
        trace = await engine.run_chain(
            chain=chain,
            initial_message=initial_message,
            environment=environment,
            scenario_id="test_scenario",
        )
        
        assert trace.status == WorkflowStatus.COMPLETED
        assert len(trace.steps) == 3
        assert trace.scenario_id == "test_scenario"
    
    @pytest.mark.asyncio
    async def test_run_chain_collects_metrics(self, mock_agents, environment, initial_message):
        metrics = MetricsCollector()
        engine = WorkflowEngine(metrics_collector=metrics)
        chain = AgentChain(mock_agents)
        
        await engine.run_chain(
            chain=chain,
            initial_message=initial_message,
            environment=environment,
        )
        
        all_metrics = metrics.get_all_metrics()
        assert len(all_metrics) == 1
        assert all_metrics[0].total_steps == 3
    
    @pytest.mark.asyncio
    async def test_run_chain_respects_max_steps(self, environment, initial_message):
        agents = [MockAgent(agent_id=f"agent_{i}") for i in range(10)]
        chain = AgentChain(agents)
        
        config = WorkflowConfig(max_steps=5)
        engine = WorkflowEngine(config=config)
        
        trace = await engine.run_chain(
            chain=chain,
            initial_message=initial_message,
            environment=environment,
        )
        
        assert len(trace.steps) == 5
        assert trace.status == WorkflowStatus.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_step_contains_agent_output(self, mock_agents, environment, initial_message):
        engine = WorkflowEngine()
        chain = AgentChain(mock_agents)
        
        trace = await engine.run_chain(
            chain=chain,
            initial_message=initial_message,
            environment=environment,
        )
        
        for i, step in enumerate(trace.steps):
            assert step.agent_id == f"agent_{i+1}"
            assert step.output_message is not None
            assert f"Response from agent {i+1}" in step.output_message.content


class TestAgentChain:
    """Tests for AgentChain."""
    
    def test_chain_length(self, mock_agents):
        chain = AgentChain(mock_agents)
        assert len(chain) == 3
    
    def test_chain_indexing(self, mock_agents):
        chain = AgentChain(mock_agents)
        assert chain[0].agent_id == "agent_1"
        assert chain[2].agent_id == "agent_3"
    
    def test_get_agent_by_id(self, mock_agents):
        chain = AgentChain(mock_agents)
        agent = chain.get_agent("agent_2")
        assert agent is not None
        assert agent.agent_id == "agent_2"
    
    def test_reset_all(self, mock_agents):
        chain = AgentChain(mock_agents)
        
        # Add some state
        for agent in chain.agents:
            agent.state.update_memory("test_key", "test_value")
        
        chain.reset_all()
        
        for agent in chain.agents:
            assert agent.state.get_memory("test_key") is None


class TestWorkflowConfig:
    """Tests for WorkflowConfig."""
    
    def test_default_values(self):
        config = WorkflowConfig()
        assert config.max_steps == 50
        assert config.timeout_seconds == 300.0
        assert config.retry_on_error is True
    
    def test_custom_values(self):
        config = WorkflowConfig(
            max_steps=100,
            timeout_seconds=600.0,
            retry_on_error=False,
        )
        assert config.max_steps == 100
        assert config.timeout_seconds == 600.0
        assert config.retry_on_error is False

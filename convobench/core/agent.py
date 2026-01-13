"""Agent base classes and state management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from convobench.core.types import (
    Action,
    ActionType,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    
    model: str = Field(description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    provider: str = Field(description="Provider name (e.g., 'openai', 'anthropic')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    system_prompt: Optional[str] = Field(default=None)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentState:
    """Internal state of an agent during workflow execution."""
    
    agent_id: str
    conversation_history: list[Message] = field(default_factory=list)
    internal_memory: dict[str, Any] = field(default_factory=dict)
    pending_tool_calls: list[ToolCall] = field(default_factory=list)
    completed_actions: list[Action] = field(default_factory=list)
    error_count: int = 0
    last_active: Optional[datetime] = None
    
    def add_message(self, message: Message) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(message)
        self.last_active = datetime.utcnow()
    
    def add_action(self, action: Action) -> None:
        """Record a completed action."""
        self.completed_actions.append(action)
        self.last_active = datetime.utcnow()
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update internal memory."""
        self.internal_memory[key] = value
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve from internal memory."""
        return self.internal_memory.get(key, default)
    
    def clear_history(self) -> None:
        """Clear conversation history but preserve memory."""
        self.conversation_history = []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "conversation_history": [m.to_dict() for m in self.conversation_history],
            "internal_memory": self.internal_memory,
            "pending_tool_calls": [t.to_dict() for t in self.pending_tool_calls],
            "completed_actions": [a.to_dict() for a in self.completed_actions],
            "error_count": self.error_count,
            "last_active": self.last_active.isoformat() if self.last_active else None,
        }


class Agent(ABC):
    """Abstract base class for agents in the benchmark."""
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        role_description: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.config = config
        self.role_description = role_description
        self.state = AgentState(agent_id=agent_id)
        self._id = uuid4()
    
    @property
    def id(self) -> UUID:
        return self._id
    
    @abstractmethod
    async def process(
        self,
        message: Message,
        available_tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[Message, list[Action]]:
        """
        Process an incoming message and return response with actions.
        
        Args:
            message: The incoming message to process
            available_tools: Tools available for this agent to use
            
        Returns:
            Tuple of (response message, list of actions taken)
        """
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Result of the tool execution
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state for a new workflow run."""
        self.state = AgentState(agent_id=self.agent_id)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        parts = []
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)
        if self.role_description:
            parts.append(f"\nYour role: {self.role_description}")
        return "\n".join(parts) if parts else ""
    
    def format_history_for_llm(self) -> list[dict[str, str]]:
        """Format conversation history for LLM API calls."""
        messages = []
        system_prompt = self.get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for msg in self.state.conversation_history:
            messages.append({
                "role": msg.role.value if msg.role != MessageRole.ENVIRONMENT else "user",
                "content": msg.content,
            })
        
        return messages
    
    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, model={self.config.model})"


class AgentChain:
    """A chain of agents that process messages sequentially."""
    
    def __init__(self, agents: list[Agent], name: str = "default"):
        self.agents = agents
        self.name = name
        self._agent_map = {a.agent_id: a for a in agents}
    
    def __len__(self) -> int:
        return len(self.agents)
    
    def __getitem__(self, index: int) -> Agent:
        return self.agents[index]
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self._agent_map.get(agent_id)
    
    def reset_all(self) -> None:
        """Reset all agents in the chain."""
        for agent in self.agents:
            agent.reset()
    
    @property
    def agent_ids(self) -> list[str]:
        return [a.agent_id for a in self.agents]


class MockAgent(Agent):
    """A mock agent for testing purposes."""
    
    def __init__(
        self,
        agent_id: str,
        responses: Optional[list[str]] = None,
        **kwargs,
    ):
        config = AgentConfig(model="mock", provider="mock")
        super().__init__(agent_id, config, **kwargs)
        self.responses = responses or ["Mock response"]
        self._response_index = 0
    
    async def process(
        self,
        message: Message,
        available_tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[Message, list[Action]]:
        self.state.add_message(message)
        
        response_text = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1
        
        response = Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            metadata={"agent_id": self.agent_id},
        )
        self.state.add_message(response)
        
        action = Action(
            action_type=ActionType.MESSAGE,
            agent_id=self.agent_id,
            payload={"content": response_text},
        )
        
        return response, [action]
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            result={"mock": "result"},
        )

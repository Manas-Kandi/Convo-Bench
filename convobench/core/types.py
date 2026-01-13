"""Core type definitions for ConvoBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ENVIRONMENT = "environment"


class ActionType(str, Enum):
    """Types of actions an agent can take."""
    TOOL_CALL = "tool_call"
    MESSAGE = "message"
    DELEGATE = "delegate"
    WAIT = "wait"
    TERMINATE = "terminate"


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Message:
    """A message in the agent conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ToolCall:
    """A tool call made by an agent."""
    tool_name: str
    arguments: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: UUID
    success: bool
    result: Any
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": str(self.tool_call_id),
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Action:
    """An action taken by an agent."""
    action_type: ActionType
    agent_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    tool_call: Optional[ToolCall] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "action_type": self.action_type.value,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WorkflowStep:
    """A single step in a workflow execution."""
    step_number: int
    agent_id: str
    input_message: Message
    output_message: Optional[Message] = None
    actions: list[Action] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    duration_ms: float = 0.0
    token_count: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "agent_id": self.agent_id,
            "input_message": self.input_message.to_dict(),
            "output_message": self.output_message.to_dict() if self.output_message else None,
            "actions": [a.to_dict() for a in self.actions],
            "tool_results": [t.to_dict() for t in self.tool_results],
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
            "error": self.error,
        }


@dataclass
class WorkflowTrace:
    """Complete trace of a workflow execution."""
    workflow_id: UUID
    scenario_id: str
    steps: list[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return sum(s.duration_ms for s in self.steps)
    
    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.steps)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": str(self.workflow_id),
            "scenario_id": self.scenario_id,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }

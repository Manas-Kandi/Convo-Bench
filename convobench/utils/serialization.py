"""Serialization utilities for ConvoBench."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from convobench.core.types import (
    Action,
    ActionType,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)


class ConvoBenchEncoder(json.JSONEncoder):
    """Custom JSON encoder for ConvoBench types."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return super().default(obj)


def serialize_trace(trace: WorkflowTrace) -> str:
    """Serialize a workflow trace to JSON string."""
    return json.dumps(trace.to_dict(), cls=ConvoBenchEncoder, indent=2)


def deserialize_trace(data: str | dict) -> WorkflowTrace:
    """Deserialize a workflow trace from JSON."""
    if isinstance(data, str):
        data = json.loads(data)
    
    steps = []
    for step_data in data.get("steps", []):
        input_msg = _deserialize_message(step_data["input_message"])
        output_msg = _deserialize_message(step_data["output_message"]) if step_data.get("output_message") else None
        
        actions = [_deserialize_action(a) for a in step_data.get("actions", [])]
        tool_results = [_deserialize_tool_result(t) for t in step_data.get("tool_results", [])]
        
        steps.append(WorkflowStep(
            step_number=step_data["step_number"],
            agent_id=step_data["agent_id"],
            input_message=input_msg,
            output_message=output_msg,
            actions=actions,
            tool_results=tool_results,
            duration_ms=step_data.get("duration_ms", 0),
            token_count=step_data.get("token_count", 0),
            error=step_data.get("error"),
        ))
    
    return WorkflowTrace(
        workflow_id=UUID(data["workflow_id"]),
        scenario_id=data["scenario_id"],
        steps=steps,
        status=WorkflowStatus(data["status"]),
        start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
        end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        metadata=data.get("metadata", {}),
    )


def _deserialize_message(data: dict) -> Message:
    """Deserialize a message."""
    return Message(
        id=UUID(data["id"]),
        role=MessageRole(data["role"]),
        content=data["content"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        metadata=data.get("metadata", {}),
    )


def _deserialize_action(data: dict) -> Action:
    """Deserialize an action."""
    tool_call = None
    if data.get("tool_call"):
        tool_call = ToolCall(
            id=UUID(data["tool_call"]["id"]),
            tool_name=data["tool_call"]["tool_name"],
            arguments=data["tool_call"]["arguments"],
            timestamp=datetime.fromisoformat(data["tool_call"]["timestamp"]),
        )
    
    return Action(
        id=UUID(data["id"]),
        action_type=ActionType(data["action_type"]),
        agent_id=data["agent_id"],
        payload=data.get("payload", {}),
        tool_call=tool_call,
        timestamp=datetime.fromisoformat(data["timestamp"]),
    )


def _deserialize_tool_result(data: dict) -> ToolResult:
    """Deserialize a tool result."""
    return ToolResult(
        tool_call_id=UUID(data["tool_call_id"]),
        success=data["success"],
        result=data["result"],
        error=data.get("error"),
        timestamp=datetime.fromisoformat(data["timestamp"]),
    )

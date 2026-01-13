"""Environment simulation for agentic workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from convobench.core.types import ToolCall, ToolResult


@dataclass
class EnvironmentState:
    """State of the simulated environment."""
    
    variables: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def set(self, key: str, value: Any) -> None:
        """Set an environment variable."""
        self.variables[key] = value
        self._log_event("set", {"key": key, "value": value})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an environment variable."""
        return self.variables.get(key, default)
    
    def add_resource(self, resource_id: str, resource: Any) -> None:
        """Add a resource to the environment."""
        self.resources[resource_id] = resource
        self._log_event("add_resource", {"resource_id": resource_id})
    
    def remove_resource(self, resource_id: str) -> Optional[Any]:
        """Remove and return a resource."""
        resource = self.resources.pop(resource_id, None)
        if resource is not None:
            self._log_event("remove_resource", {"resource_id": resource_id})
        return resource
    
    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event."""
        self.event_log.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": self.variables,
            "resources": list(self.resources.keys()),
            "event_count": len(self.event_log),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Tool:
    """Definition of a tool available in the environment."""
    
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], EnvironmentState], Any]
    requires_confirmation: bool = False
    side_effects: list[str] = field(default_factory=list)
    
    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Environment:
    """
    Simulated environment for agent workflows.
    
    Provides tools, state management, and feedback mechanisms
    that agents interact with during benchmark execution.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.state = EnvironmentState()
        self._tools: dict[str, Tool] = {}
        self._id = uuid4()
        self._setup_default_tools()
    
    @property
    def id(self) -> UUID:
        return self._id
    
    def _setup_default_tools(self) -> None:
        """Set up default tools available in all environments."""
        self.register_tool(Tool(
            name="read_variable",
            description="Read a variable from the environment state",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Variable name to read"},
                },
                "required": ["key"],
            },
            handler=lambda args, state: state.get(args["key"]),
        ))
        
        self.register_tool(Tool(
            name="write_variable",
            description="Write a variable to the environment state",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Variable name"},
                    "value": {"type": "string", "description": "Value to write"},
                },
                "required": ["key", "value"],
            },
            handler=lambda args, state: state.set(args["key"], args["value"]),
            side_effects=["modifies_state"],
        ))
        
        self.register_tool(Tool(
            name="list_resources",
            description="List all available resources in the environment",
            parameters={"type": "object", "properties": {}},
            handler=lambda args, state: list(state.resources.keys()),
        ))
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the environment."""
        self._tools[tool.name] = tool
    
    def unregister_tool(self, tool_name: str) -> None:
        """Remove a tool from the environment."""
        self._tools.pop(tool_name, None)
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get schemas for all available tools."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    def get_tool_names(self) -> list[str]:
        """Get names of all available tools."""
        return list(self._tools.keys())
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call in the environment.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Result of the tool execution
        """
        tool = self._tools.get(tool_call.tool_name)
        
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_call.tool_name}",
            )
        
        try:
            result = tool.handler(tool_call.arguments, self.state)
            self.state._log_event("tool_execution", {
                "tool": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": True,
            })
            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                result=result,
            )
        except Exception as e:
            self.state._log_event("tool_execution", {
                "tool": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": False,
                "error": str(e),
            })
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                result=None,
                error=str(e),
            )
    
    def reset(self) -> None:
        """Reset environment to initial state."""
        self.state = EnvironmentState()
    
    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot of current environment state."""
        return {
            "id": str(self._id),
            "name": self.name,
            "state": self.state.to_dict(),
            "tools": self.get_tool_names(),
        }
    
    def inject_state(self, variables: dict[str, Any]) -> None:
        """Inject variables into environment state."""
        for key, value in variables.items():
            self.state.set(key, value)


class ScenarioEnvironment(Environment):
    """
    Environment configured for a specific benchmark scenario.
    
    Extends base environment with scenario-specific tools,
    initial state, and validation logic.
    """
    
    def __init__(
        self,
        name: str,
        initial_state: Optional[dict[str, Any]] = None,
        custom_tools: Optional[list[Tool]] = None,
        constraints: Optional[list[str]] = None,
    ):
        super().__init__(name)
        self.constraints = constraints or []
        self._initial_state = initial_state or {}
        
        if initial_state:
            self.inject_state(initial_state)
        
        if custom_tools:
            for tool in custom_tools:
                self.register_tool(tool)
    
    def validate_constraints(self) -> list[str]:
        """
        Validate that environment constraints are satisfied.
        
        Returns:
            List of constraint violations (empty if all satisfied)
        """
        violations = []
        for constraint in self.constraints:
            if not self._check_constraint(constraint):
                violations.append(constraint)
        return violations
    
    def _check_constraint(self, constraint: str) -> bool:
        """Check a single constraint. Override for custom logic."""
        return True
    
    def reset(self) -> None:
        """Reset to initial scenario state."""
        super().reset()
        if self._initial_state:
            self.inject_state(self._initial_state)

"""Base adapter class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from convobench.core.agent import Agent, AgentConfig, AgentState
from convobench.core.types import Action, ActionType, Message, MessageRole, ToolCall, ToolResult


class AdapterConfig(BaseModel):
    """Configuration for an LLM adapter."""
    
    api_key: Optional[str] = Field(default=None, description="API key (uses env var if not set)")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class BaseAdapter(Agent, ABC):
    """
    Base adapter for connecting to LLM providers.
    
    Adapters translate between ConvoBench's agent interface
    and provider-specific APIs.
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        adapter_config: Optional[AdapterConfig] = None,
        role_description: Optional[str] = None,
    ):
        super().__init__(agent_id, config, role_description)
        self.adapter_config = adapter_config or AdapterConfig()
        self._client = None
    
    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Make API call to the provider."""
        pass
    
    async def process(
        self,
        message: Message,
        available_tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[Message, list[Action]]:
        """Process a message and return response with actions."""
        if self._client is None:
            await self._initialize_client()
        
        self.state.add_message(message)
        
        messages = self.format_history_for_llm()
        
        response = await self._call_api(messages, available_tools)
        
        response_message, actions = self._parse_response(response)
        
        self.state.add_message(response_message)
        for action in actions:
            self.state.add_action(action)
        
        return response_message, actions
    
    def _parse_response(self, response: dict[str, Any]) -> tuple[Message, list[Action]]:
        """Parse provider response into Message and Actions."""
        content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        
        response_message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "agent_id": self.agent_id,
                "model": self.config.model,
                "usage": response.get("usage", {}),
            },
        )
        
        actions = []
        
        if content:
            actions.append(Action(
                action_type=ActionType.MESSAGE,
                agent_id=self.agent_id,
                payload={"content": content},
            ))
        
        for tc in tool_calls:
            tool_call = ToolCall(
                tool_name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            )
            actions.append(Action(
                action_type=ActionType.TOOL_CALL,
                agent_id=self.agent_id,
                tool_call=tool_call,
                payload={"tool_name": tc.get("name")},
            ))
        
        return response_message, actions
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.
        
        Note: Actual tool execution is handled by the Environment.
        This method is for agent-internal tool handling if needed.
        """
        return ToolResult(
            tool_call_id=tool_call.id,
            success=False,
            result=None,
            error="Tool execution should be handled by Environment",
        )
    
    def _format_tools_for_api(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for the specific provider API. Override if needed."""
        return tools

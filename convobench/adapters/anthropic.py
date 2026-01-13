"""Anthropic adapter for ConvoBench."""

from __future__ import annotations

import json
from typing import Any, Optional

from convobench.adapters.base import AdapterConfig, BaseAdapter
from convobench.core.agent import AgentConfig


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic models (Claude 3, etc.)."""
    
    def __init__(
        self,
        agent_id: str,
        model: str = "claude-3-opus-20240229",
        adapter_config: Optional[AdapterConfig] = None,
        role_description: Optional[str] = None,
        **kwargs,
    ):
        config = AgentConfig(
            model=model,
            provider="anthropic",
            **kwargs,
        )
        super().__init__(agent_id, config, adapter_config, role_description)
    
    async def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            
            client_kwargs = {}
            if self.adapter_config.api_key:
                client_kwargs["api_key"] = self.adapter_config.api_key
            if self.adapter_config.base_url:
                client_kwargs["base_url"] = self.adapter_config.base_url
            
            self._client = AsyncAnthropic(**client_kwargs)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Call Anthropic API."""
        system_message = None
        api_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                api_messages.append(msg)
        
        kwargs = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        if tools:
            kwargs["tools"] = self._format_tools_for_api(tools)
        
        response = await self._client.messages.create(**kwargs)
        
        return self._parse_anthropic_response(response)
    
    def _parse_anthropic_response(self, response) -> dict[str, Any]:
        """Parse Anthropic response format."""
        result = {
            "content": "",
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        
        return result
    
    def _format_tools_for_api(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for Anthropic API."""
        formatted = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                formatted.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            else:
                formatted.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {}),
                })
        return formatted

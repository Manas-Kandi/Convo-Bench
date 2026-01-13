"""OpenAI adapter for ConvoBench."""

from __future__ import annotations

import json
from typing import Any, Optional

from convobench.adapters.base import AdapterConfig, BaseAdapter
from convobench.core.agent import AgentConfig


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models (GPT-4, GPT-3.5, etc.)."""
    
    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4",
        adapter_config: Optional[AdapterConfig] = None,
        role_description: Optional[str] = None,
        **kwargs,
    ):
        config = AgentConfig(
            model=model,
            provider="openai",
            **kwargs,
        )
        super().__init__(agent_id, config, adapter_config, role_description)
    
    async def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            client_kwargs = {}
            if self.adapter_config.api_key:
                client_kwargs["api_key"] = self.adapter_config.api_key
            if self.adapter_config.base_url:
                client_kwargs["base_url"] = self.adapter_config.base_url
            
            self._client = AsyncOpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Call OpenAI API."""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if tools:
            kwargs["tools"] = self._format_tools_for_api(tools)
            kwargs["tool_choice"] = "auto"
        
        response = await self._client.chat.completions.create(**kwargs)
        
        return self._parse_openai_response(response)
    
    def _parse_openai_response(self, response) -> dict[str, Any]:
        """Parse OpenAI response format."""
        message = response.choices[0].message
        
        result = {
            "content": message.content or "",
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}
                
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": arguments,
                })
        
        return result
    
    def _format_tools_for_api(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for OpenAI API."""
        formatted = []
        for tool in tools:
            if "function" in tool:
                formatted.append(tool)
            else:
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })
        return formatted

"""NVIDIA NIM adapter for ConvoBench.

Supports multiple models via NVIDIA's OpenAI-compatible API:
- nvidia/nemotron-3-nano-30b-a3b (with reasoning/thinking support)
- moonshotai/kimi-k2-thinking
- mistralai/mistral-large-3-675b-instruct-2512
- minimaxai/minimax-m2
- qwen/qwen3-next-80b-a3b-instruct
- tiiuae/falcon3-7b-instruct
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from convobench.adapters.base import AdapterConfig, BaseAdapter

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
from convobench.core.agent import AgentConfig


# Model configurations with recommended parameters
NVIDIA_MODELS = {
    "nemotron-nano": {
        "model_id": "nvidia/nemotron-3-nano-30b-a3b",
        "max_tokens": 16384,
        "temperature": 1.0,
        "top_p": 1.0,
        "supports_reasoning": True,
    },
    "kimi-k2": {
        "model_id": "moonshotai/kimi-k2-thinking",
        "max_tokens": 16384,
        "temperature": 1.0,
        "top_p": 1.0,
        "supports_reasoning": True,
    },
    "mistral-large": {
        "model_id": "mistralai/mistral-large-3-675b-instruct-2512",
        "max_tokens": 2048,
        "temperature": 0.15,
        "top_p": 1.0,
        "supports_reasoning": False,
    },
    "minimax-m2": {
        "model_id": "minimaxai/minimax-m2",
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "supports_reasoning": False,
    },
    "qwen3-next": {
        "model_id": "qwen/qwen3-next-80b-a3b-instruct",
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.7,
        "supports_reasoning": False,
    },
    "falcon3": {
        "model_id": "tiiuae/falcon3-7b-instruct",
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.7,
        "supports_reasoning": False,
    },
}


class NVIDIAAdapterConfig(AdapterConfig):
    """Configuration specific to NVIDIA NIM adapter."""
    
    enable_thinking: bool = True
    reasoning_budget: int = 16384
    stream: bool = False  # Set to False for benchmark (easier to process)


class NVIDIAAdapter(BaseAdapter):
    """
    Adapter for NVIDIA NIM models via OpenAI-compatible API.
    
    Supports multiple models including those with reasoning/thinking capabilities.
    Uses https://integrate.api.nvidia.com/v1 as the base URL.
    """
    
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    
    def __init__(
        self,
        agent_id: str,
        model: str = "nemotron-nano",
        adapter_config: Optional[NVIDIAAdapterConfig] = None,
        role_description: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize NVIDIA adapter.
        
        Args:
            agent_id: Unique identifier for this agent
            model: Model shortname (e.g., 'nemotron-nano', 'mistral-large') 
                   or full model ID (e.g., 'nvidia/nemotron-3-nano-30b-a3b')
            adapter_config: NVIDIA-specific configuration
            role_description: Role description for the agent
            **kwargs: Additional AgentConfig parameters
        """
        # Resolve model configuration
        if model in NVIDIA_MODELS:
            model_config = NVIDIA_MODELS[model]
            model_id = model_config["model_id"]
            default_max_tokens = model_config["max_tokens"]
            default_temp = model_config["temperature"]
            default_top_p = model_config["top_p"]
            self._supports_reasoning = model_config["supports_reasoning"]
        else:
            # Assume full model ID provided
            model_id = model
            default_max_tokens = 4096
            default_temp = 0.7
            default_top_p = 1.0
            self._supports_reasoning = False
        
        # Use provided values or defaults
        config = AgentConfig(
            model=model_id,
            provider="nvidia",
            temperature=kwargs.pop("temperature", default_temp),
            max_tokens=kwargs.pop("max_tokens", default_max_tokens),
            **kwargs,
        )
        
        self._top_p = default_top_p
        self._nvidia_config = adapter_config or NVIDIAAdapterConfig()
        
        # Ensure base_url is set for NVIDIA
        if self._nvidia_config.base_url is None:
            self._nvidia_config.base_url = self.NVIDIA_BASE_URL
        
        super().__init__(agent_id, config, self._nvidia_config, role_description)
    
    async def _initialize_client(self) -> None:
        """Initialize OpenAI client with NVIDIA endpoint."""
        try:
            from openai import AsyncOpenAI
            
            api_key = self._nvidia_config.api_key or os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError(
                    "NVIDIA API key required. Set NVIDIA_API_KEY environment variable "
                    "or pass api_key in adapter_config."
                )
            
            self._client = AsyncOpenAI(
                base_url=self._nvidia_config.base_url,
                api_key=api_key,
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Call NVIDIA NIM API."""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self._top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,  # Non-streaming for easier processing
        }
        
        # Add reasoning support for compatible models
        if self._supports_reasoning and self._nvidia_config.enable_thinking:
            kwargs["extra_body"] = {
                "reasoning_budget": self._nvidia_config.reasoning_budget,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        
        # Add tools if provided (not all NVIDIA models support tools)
        if tools:
            kwargs["tools"] = self._format_tools_for_api(tools)
            kwargs["tool_choice"] = "auto"
        
        response = await self._client.chat.completions.create(**kwargs)
        
        return self._parse_nvidia_response(response)
    
    def _parse_nvidia_response(self, response) -> dict[str, Any]:
        """Parse NVIDIA API response."""
        message = response.choices[0].message
        
        result = {
            "content": message.content or "",
            "tool_calls": [],
            "usage": {},
        }
        
        # Extract usage if available
        if hasattr(response, "usage") and response.usage:
            result["usage"] = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        
        # Extract reasoning content if present (for thinking models)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            result["reasoning"] = message.reasoning_content
        
        # Extract tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
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
        """Format tools for NVIDIA API (OpenAI-compatible format)."""
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


def create_nvidia_agents(
    num_agents: int,
    model: str = "nemotron-nano",
    role_descriptions: Optional[list[str]] = None,
    **kwargs,
) -> list[NVIDIAAdapter]:
    """
    Create multiple NVIDIA agents for benchmarking.
    
    Args:
        num_agents: Number of agents to create
        model: Model shortname or full ID
        role_descriptions: Optional role descriptions for each agent
        **kwargs: Additional configuration
        
    Returns:
        List of NVIDIAAdapter instances
    """
    agents = []
    for i in range(num_agents):
        role = None
        if role_descriptions and i < len(role_descriptions):
            role = role_descriptions[i]
        
        agents.append(NVIDIAAdapter(
            agent_id=f"nvidia_agent_{i+1}",
            model=model,
            role_description=role,
            **kwargs,
        ))
    
    return agents


def create_mixed_nvidia_agents(
    models: list[str],
    role_descriptions: Optional[list[str]] = None,
    **kwargs,
) -> list[NVIDIAAdapter]:
    """
    Create agents with different NVIDIA models for comparison.
    
    Args:
        models: List of model shortnames or IDs
        role_descriptions: Optional role descriptions
        **kwargs: Additional configuration
        
    Returns:
        List of NVIDIAAdapter instances with different models
    """
    agents = []
    for i, model in enumerate(models):
        role = None
        if role_descriptions and i < len(role_descriptions):
            role = role_descriptions[i]
        
        agents.append(NVIDIAAdapter(
            agent_id=f"nvidia_{model.replace('/', '_')}_{i+1}",
            model=model,
            role_description=role,
            **kwargs,
        ))
    
    return agents

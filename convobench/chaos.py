"""Operational chaos injection for environments.

Provides a wrapper Environment that can inject:
- latency
- rate limits
- random failures/timeouts
- permission errors

This is designed to make scenarios realistic without modifying every Tool.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Optional

from convobench.core.environment import Environment
from convobench.core.types import ToolCall, ToolResult


@dataclass
class ChaosConfig:
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    permission_error_rate: float = 0.0
    min_latency_ms: int = 0
    max_latency_ms: int = 0

    # Simple rate limit: max tool calls per window
    rate_limit_calls: Optional[int] = None
    rate_limit_window_ms: int = 1000

    seed: Optional[int] = None


class ChaosEnvironment(Environment):
    def __init__(self, inner: Environment, config: ChaosConfig):
        super().__init__(name=f"chaos({inner.name})")
        self._inner = inner
        self._config = config
        self._rng = random.Random(config.seed)
        self._window_start_ms = 0.0
        self._window_calls = 0

    def get_available_tools(self) -> list[dict[str, Any]]:
        return self._inner.get_available_tools()

    def get_tool_names(self) -> list[str]:
        return self._inner.get_tool_names()

    def snapshot(self) -> dict[str, Any]:
        snap = self._inner.snapshot()
        snap["chaos"] = {
            "failure_rate": self._config.failure_rate,
            "timeout_rate": self._config.timeout_rate,
            "permission_error_rate": self._config.permission_error_rate,
            "min_latency_ms": self._config.min_latency_ms,
            "max_latency_ms": self._config.max_latency_ms,
            "rate_limit_calls": self._config.rate_limit_calls,
            "rate_limit_window_ms": self._config.rate_limit_window_ms,
            "seed": self._config.seed,
        }
        return snap

    def reset(self) -> None:
        self._inner.reset()
        self._window_start_ms = 0.0
        self._window_calls = 0

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        # Rate limiting
        now = asyncio.get_event_loop().time() * 1000
        if self._window_start_ms == 0.0:
            self._window_start_ms = now
        if now - self._window_start_ms > self._config.rate_limit_window_ms:
            self._window_start_ms = now
            self._window_calls = 0

        if self._config.rate_limit_calls is not None:
            if self._window_calls >= self._config.rate_limit_calls:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    result=None,
                    error="rate_limited",
                )
            self._window_calls += 1

        # Latency
        if self._config.max_latency_ms > 0:
            delay = self._rng.randint(self._config.min_latency_ms, self._config.max_latency_ms)
            await asyncio.sleep(delay / 1000)

        # Fault injection
        r = self._rng.random()
        if self._config.permission_error_rate > 0 and r < self._config.permission_error_rate:
            return ToolResult(tool_call_id=tool_call.id, success=False, result=None, error="permission_denied")

        r = self._rng.random()
        if self._config.timeout_rate > 0 and r < self._config.timeout_rate:
            return ToolResult(tool_call_id=tool_call.id, success=False, result=None, error="timeout")

        r = self._rng.random()
        if self._config.failure_rate > 0 and r < self._config.failure_rate:
            return ToolResult(tool_call_id=tool_call.id, success=False, result=None, error="tool_failure")

        return await self._inner.execute_tool(tool_call)

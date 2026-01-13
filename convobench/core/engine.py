"""Workflow execution engine for running benchmark scenarios."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from convobench.core.agent import Agent, AgentChain
from convobench.core.environment import Environment
from convobench.core.metrics import MetricsCollector
from convobench.core.types import (
    Action,
    ActionType,
    Message,
    MessageRole,
    ToolCall,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    
    max_steps: int = 50
    timeout_seconds: float = 300.0
    allow_parallel_agents: bool = False
    retry_on_error: bool = True
    max_retries: int = 3
    step_delay_ms: float = 0.0


class WorkflowEngine:
    """
    Engine for executing multi-agent workflows.
    
    Orchestrates agent interactions, manages environment state,
    and collects execution traces for evaluation.
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.config = config or WorkflowConfig()
        self.metrics = metrics_collector or MetricsCollector()
        self._current_trace: Optional[WorkflowTrace] = None
    
    async def run_chain(
        self,
        chain: AgentChain,
        initial_message: Message,
        environment: Environment,
        scenario_id: str = "default",
    ) -> WorkflowTrace:
        """
        Execute a workflow through an agent chain.
        
        Each agent processes the output of the previous agent,
        simulating information passing through intermediaries.
        
        Args:
            chain: Chain of agents to execute
            initial_message: Starting message for the workflow
            environment: Environment for tool execution
            scenario_id: Identifier for the scenario
            
        Returns:
            Complete workflow trace
        """
        trace = WorkflowTrace(
            workflow_id=uuid4(),
            scenario_id=scenario_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.utcnow(),
        )
        self._current_trace = trace
        
        current_message = initial_message
        step_number = 0
        
        try:
            for agent in chain.agents:
                if step_number >= self.config.max_steps:
                    trace.status = WorkflowStatus.TIMEOUT
                    break
                
                step = await self._execute_step(
                    agent=agent,
                    message=current_message,
                    environment=environment,
                    step_number=step_number,
                )
                trace.steps.append(step)
                
                if step.error and not self.config.retry_on_error:
                    trace.status = WorkflowStatus.FAILED
                    break
                
                if step.output_message:
                    current_message = step.output_message
                
                step_number += 1
                
                if self.config.step_delay_ms > 0:
                    await asyncio.sleep(self.config.step_delay_ms / 1000)
            
            if trace.status == WorkflowStatus.RUNNING:
                trace.status = WorkflowStatus.COMPLETED
                
        except asyncio.TimeoutError:
            trace.status = WorkflowStatus.TIMEOUT
        except Exception as e:
            trace.status = WorkflowStatus.FAILED
            trace.metadata["error"] = str(e)
        finally:
            trace.end_time = datetime.utcnow()
            self._current_trace = None
        
        self.metrics.collect_from_trace(trace)
        return trace
    
    async def run_collaborative(
        self,
        agents: list[Agent],
        initial_message: Message,
        environment: Environment,
        scenario_id: str = "default",
        coordination_strategy: str = "round_robin",
    ) -> WorkflowTrace:
        """
        Execute a collaborative workflow where agents work together.
        
        Unlike chain execution, agents can interact multiple times
        and coordinate on shared goals.
        
        Args:
            agents: List of agents participating
            initial_message: Starting message/goal
            environment: Shared environment
            scenario_id: Identifier for the scenario
            coordination_strategy: How to coordinate agents
            
        Returns:
            Complete workflow trace
        """
        trace = WorkflowTrace(
            workflow_id=uuid4(),
            scenario_id=scenario_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata={"coordination_strategy": coordination_strategy},
        )
        self._current_trace = trace
        
        try:
            if coordination_strategy == "round_robin":
                await self._run_round_robin(agents, initial_message, environment, trace)
            elif coordination_strategy == "broadcast":
                await self._run_broadcast(agents, initial_message, environment, trace)
            elif coordination_strategy == "hierarchical":
                await self._run_hierarchical(agents, initial_message, environment, trace)
            else:
                raise ValueError(f"Unknown coordination strategy: {coordination_strategy}")
            
            if trace.status == WorkflowStatus.RUNNING:
                trace.status = WorkflowStatus.COMPLETED
                
        except asyncio.TimeoutError:
            trace.status = WorkflowStatus.TIMEOUT
        except Exception as e:
            trace.status = WorkflowStatus.FAILED
            trace.metadata["error"] = str(e)
        finally:
            trace.end_time = datetime.utcnow()
            self._current_trace = None
        
        self.metrics.collect_from_trace(trace)
        return trace
    
    async def _execute_step(
        self,
        agent: Agent,
        message: Message,
        environment: Environment,
        step_number: int,
    ) -> WorkflowStep:
        """Execute a single workflow step."""
        start_time = time.perf_counter()
        
        step = WorkflowStep(
            step_number=step_number,
            agent_id=agent.agent_id,
            input_message=message,
        )
        
        try:
            response, actions = await agent.process(
                message=message,
                available_tools=environment.get_available_tools(),
            )
            step.output_message = response
            step.actions = actions
            
            # Execute any tool calls
            for action in actions:
                if action.action_type == ActionType.TOOL_CALL and action.tool_call:
                    result = await environment.execute_tool(action.tool_call)
                    step.tool_results.append(result)
            
            # Estimate token count (simplified)
            step.token_count = self._estimate_tokens(message, response)
            
        except Exception as e:
            step.error = str(e)
            agent.state.error_count += 1
        
        step.duration_ms = (time.perf_counter() - start_time) * 1000
        return step
    
    async def _run_round_robin(
        self,
        agents: list[Agent],
        initial_message: Message,
        environment: Environment,
        trace: WorkflowTrace,
    ) -> None:
        """Round-robin coordination: agents take turns."""
        current_message = initial_message
        step_number = 0
        agent_index = 0
        
        # Continue until termination or max steps
        while step_number < self.config.max_steps:
            agent = agents[agent_index % len(agents)]
            
            step = await self._execute_step(
                agent=agent,
                message=current_message,
                environment=environment,
                step_number=step_number,
            )
            trace.steps.append(step)
            
            # Check for termination action
            for action in step.actions:
                if action.action_type == ActionType.TERMINATE:
                    return
            
            if step.output_message:
                current_message = step.output_message
            
            step_number += 1
            agent_index += 1
    
    async def _run_broadcast(
        self,
        agents: list[Agent],
        initial_message: Message,
        environment: Environment,
        trace: WorkflowTrace,
    ) -> None:
        """Broadcast coordination: all agents receive same message."""
        current_message = initial_message
        step_number = 0
        
        while step_number < self.config.max_steps:
            # All agents process in parallel
            tasks = [
                self._execute_step(agent, current_message, environment, step_number + i)
                for i, agent in enumerate(agents)
            ]
            steps = await asyncio.gather(*tasks)
            trace.steps.extend(steps)
            
            # Aggregate responses for next round
            responses = [s.output_message for s in steps if s.output_message]
            if not responses:
                break
            
            # Create aggregated message for next round
            aggregated_content = "\n---\n".join([
                f"[{s.agent_id}]: {s.output_message.content}"
                for s in steps if s.output_message
            ])
            current_message = Message(
                role=MessageRole.ENVIRONMENT,
                content=f"Responses from all agents:\n{aggregated_content}",
            )
            
            step_number += len(agents)
            
            # Check if any agent wants to terminate
            for step in steps:
                for action in step.actions:
                    if action.action_type == ActionType.TERMINATE:
                        return
    
    async def _run_hierarchical(
        self,
        agents: list[Agent],
        initial_message: Message,
        environment: Environment,
        trace: WorkflowTrace,
    ) -> None:
        """Hierarchical coordination: first agent delegates to others."""
        if not agents:
            return
        
        coordinator = agents[0]
        workers = agents[1:] if len(agents) > 1 else []
        
        current_message = initial_message
        step_number = 0
        
        while step_number < self.config.max_steps:
            # Coordinator decides what to do
            coord_step = await self._execute_step(
                agent=coordinator,
                message=current_message,
                environment=environment,
                step_number=step_number,
            )
            trace.steps.append(coord_step)
            step_number += 1
            
            # Check for delegation actions
            delegations = [
                a for a in coord_step.actions
                if a.action_type == ActionType.DELEGATE
            ]
            
            if delegations and workers:
                # Execute delegated tasks
                for delegation in delegations:
                    target_agent_id = delegation.payload.get("target_agent")
                    task_message = Message(
                        role=MessageRole.USER,
                        content=delegation.payload.get("task", ""),
                    )
                    
                    # Find target worker or use round-robin
                    worker = next(
                        (w for w in workers if w.agent_id == target_agent_id),
                        workers[step_number % len(workers)]
                    )
                    
                    worker_step = await self._execute_step(
                        agent=worker,
                        message=task_message,
                        environment=environment,
                        step_number=step_number,
                    )
                    trace.steps.append(worker_step)
                    step_number += 1
                    
                    # Feed result back to coordinator
                    if worker_step.output_message:
                        current_message = Message(
                            role=MessageRole.ENVIRONMENT,
                            content=f"Result from {worker.agent_id}: {worker_step.output_message.content}",
                        )
            
            # Check for termination
            for action in coord_step.actions:
                if action.action_type == ActionType.TERMINATE:
                    return
            
            if coord_step.output_message and not delegations:
                current_message = coord_step.output_message
    
    def _estimate_tokens(self, input_msg: Message, output_msg: Message) -> int:
        """Rough token estimation (4 chars per token)."""
        input_len = len(input_msg.content)
        output_len = len(output_msg.content) if output_msg else 0
        return (input_len + output_len) // 4
    
    def get_current_trace(self) -> Optional[WorkflowTrace]:
        """Get the currently executing trace, if any."""
        return self._current_trace

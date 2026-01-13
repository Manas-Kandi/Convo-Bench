"""FastAPI backend for ConvoBench frontend."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed, rely on environment variables

from convobench.adapters import NVIDIA_MODELS, create_nvidia_agents
from convobench.adapters.nvidia import NVIDIAAdapter
from convobench.bench import BenchmarkConfig, create_mock_agents
from convobench.core.agent import MockAgent
from convobench.core.engine import WorkflowConfig, WorkflowEngine
from convobench.core.metrics import MetricsCollector
from convobench.core.types import WorkflowStatus
from convobench.evaluation.evaluator import EvaluatorConfig, ExternalEvaluator
from convobench.scenarios import (
    AdversarialRelay,
    CollaborativePlanning,
    ConstrainedRelay,
    ConstraintViolation,
    ErrorInjection,
    GoalDecomposition,
    HandoffScenario,
    InformationRelay,
    NoisyRelay,
    ResourceAllocation,
    StateSynchronization,
    ToolCoordination,
)
import re


def calculate_preservation_metrics(original: str, output: str, step_number: int) -> dict:
    """Calculate information preservation metrics between original and output."""
    
    # Extract key elements from original
    # Numbers (dates, amounts, counts)
    original_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?::\d+)?\b', original))
    output_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?::\d+)?\b', output))
    
    # Key terms (capitalized words, quoted strings)
    original_terms = set(re.findall(r'"([^"]+)"', original))
    output_terms = set(re.findall(r'"([^"]+)"', output))
    
    # Calculate preservation rates
    number_preserved = len(original_numbers & output_numbers)
    number_total = len(original_numbers)
    number_rate = number_preserved / number_total if number_total > 0 else 1.0
    
    term_preserved = len(original_terms & output_terms)
    term_total = len(original_terms)
    term_rate = term_preserved / term_total if term_total > 0 else 1.0
    
    # Content length ratio (detect significant truncation)
    length_ratio = len(output) / len(original) if len(original) > 0 else 0
    
    # Overall preservation score
    overall = (number_rate * 0.4 + term_rate * 0.4 + min(length_ratio, 1.0) * 0.2)
    
    return {
        "step": step_number,
        "numbers_preserved": f"{number_preserved}/{number_total}",
        "terms_preserved": f"{term_preserved}/{term_total}",
        "length_ratio": round(length_ratio, 2),
        "preservation_score": round(overall * 100, 1),
    }

app = FastAPI(title="ConvoBench API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active runs
active_runs: dict[str, dict] = {}
websocket_connections: dict[str, WebSocket] = {}


# Request/Response Models
class ScenarioConfig(BaseModel):
    scenario_type: str
    chain_length: int = 3
    complexity: str = "medium"
    num_constraints: int = 3
    num_agents: int = 3


class RunConfig(BaseModel):
    scenario: ScenarioConfig
    model: str = "mock"
    num_runs: int = 5
    evaluate: bool = True


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


# Scenario factory
SCENARIO_MAP = {
    "information_relay": lambda cfg: InformationRelay(
        chain_length=cfg.chain_length,
        message_complexity=cfg.complexity,
    ),
    "constrained_relay": lambda cfg: ConstrainedRelay(
        chain_length=cfg.chain_length,
        num_constraints=cfg.num_constraints,
    ),
    "noisy_relay": lambda cfg: NoisyRelay(
        chain_length=cfg.chain_length,
        noise_level=cfg.complexity,
    ),
    "collaborative_planning": lambda cfg: CollaborativePlanning(
        num_agents=cfg.num_agents,
        planning_domain="project",
    ),
    "goal_decomposition": lambda cfg: GoalDecomposition(
        chain_length=cfg.chain_length,
        goal_complexity=cfg.complexity,
    ),
    "resource_allocation": lambda cfg: ResourceAllocation(
        num_stakeholders=cfg.num_agents,
        resource_type="budget",
    ),
    "tool_coordination": lambda cfg: ToolCoordination(
        num_agents=cfg.num_agents,
        task_type="data_pipeline",
    ),
    "state_synchronization": lambda cfg: StateSynchronization(
        num_agents=cfg.num_agents,
        state_complexity=cfg.complexity,
    ),
    "handoff_scenario": lambda cfg: HandoffScenario(
        num_handoffs=cfg.chain_length,
        task_domain="support",
    ),
    "adversarial_relay": lambda cfg: AdversarialRelay(
        chain_length=cfg.chain_length,
        attack_type="subtle",
    ),
    "constraint_violation": lambda cfg: ConstraintViolation(
        num_agents=cfg.num_agents,
        violation_type="budget",
    ),
    "error_injection": lambda cfg: ErrorInjection(
        chain_length=cfg.chain_length,
        error_type="tool_failure",
    ),
}


@app.get("/")
async def root():
    return {"status": "ok", "service": "ConvoBench API"}


@app.get("/scenarios")
async def list_scenarios():
    """List all available scenarios."""
    scenarios = [
        {
            "id": "information_relay",
            "name": "Information Relay",
            "category": "relay",
            "description": "Pass information through agent chains",
        },
        {
            "id": "constrained_relay",
            "name": "Constrained Relay",
            "category": "relay",
            "description": "Relay with constraint preservation",
        },
        {
            "id": "noisy_relay",
            "name": "Noisy Relay",
            "category": "relay",
            "description": "Filter noise while preserving signal",
        },
        {
            "id": "collaborative_planning",
            "name": "Collaborative Planning",
            "category": "planning",
            "description": "Multi-agent collaborative planning",
        },
        {
            "id": "goal_decomposition",
            "name": "Goal Decomposition",
            "category": "planning",
            "description": "Break down goals into subtasks",
        },
        {
            "id": "resource_allocation",
            "name": "Resource Allocation",
            "category": "planning",
            "description": "Allocate limited resources",
        },
        {
            "id": "tool_coordination",
            "name": "Tool Coordination",
            "category": "coordination",
            "description": "Coordinate tool usage",
        },
        {
            "id": "state_synchronization",
            "name": "State Synchronization",
            "category": "coordination",
            "description": "Maintain shared state",
        },
        {
            "id": "handoff_scenario",
            "name": "Handoff Scenario",
            "category": "coordination",
            "description": "Task handoffs between agents",
        },
        {
            "id": "adversarial_relay",
            "name": "Adversarial Relay",
            "category": "adversarial",
            "description": "Detect adversarial modifications",
        },
        {
            "id": "constraint_violation",
            "name": "Constraint Violation",
            "category": "adversarial",
            "description": "Test constraint adherence",
        },
        {
            "id": "error_injection",
            "name": "Error Injection",
            "category": "adversarial",
            "description": "Handle injected errors",
        },
    ]
    return {"scenarios": scenarios}


@app.get("/models")
async def list_models():
    """List available models."""
    models = [{"id": "mock", "name": "Mock Agent", "provider": "mock", "available": True}]
    
    # Check if NVIDIA API key is set
    nvidia_available = bool(os.environ.get("NVIDIA_API_KEY"))
    
    for shortname, config in NVIDIA_MODELS.items():
        models.append({
            "id": shortname,
            "name": config["model_id"].split("/")[-1],
            "provider": "nvidia",
            "full_id": config["model_id"],
            "supports_reasoning": config.get("supports_reasoning", False),
            "available": nvidia_available,
        })
    
    return {"models": models, "nvidia_configured": nvidia_available}


@app.post("/runs", response_model=RunResponse)
async def create_run(config: RunConfig):
    """Start a new benchmark run."""
    run_id = f"run_{uuid4().hex[:12]}"
    
    # Validate scenario
    if config.scenario.scenario_type not in SCENARIO_MAP:
        raise HTTPException(400, f"Unknown scenario: {config.scenario.scenario_type}")
    
    # Store run state
    active_runs[run_id] = {
        "id": run_id,
        "config": config.model_dump(),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "traces": [],
        "evaluations": [],
        "messages": [],
        "progress": 0,
    }
    
    # Start run in background
    asyncio.create_task(execute_run(run_id, config))
    
    return RunResponse(run_id=run_id, status="started", message="Run initiated")


async def execute_run(run_id: str, config: RunConfig):
    """Execute a benchmark run."""
    run_state = active_runs[run_id]
    run_state["status"] = "running"
    
    # Wait for WebSocket to connect
    await asyncio.sleep(0.5)
    
    try:
        # Create scenario
        scenario = SCENARIO_MAP[config.scenario.scenario_type](config.scenario)
        instance = scenario.create_instance()
        
        # Create agents with proper relay-focused system prompts
        num_agents = config.scenario.chain_length
        
        # Role descriptions that encourage information preservation without meta-awareness
        relay_roles = [
            "You are a detail-oriented assistant. When given information, your job is to understand it fully and communicate it clearly to others. Preserve all specific details like names, numbers, dates, and requirements.",
            "You are a communications specialist. Your role is to receive information and pass it along accurately. Focus on maintaining the integrity of all details, especially numerical values and specific requirements.",
            "You are a project coordinator. When you receive task information, summarize it clearly while preserving ALL critical details. Do not take action - just prepare the information for handoff.",
            "You are an executive assistant. Your job is to process information and prepare clear summaries. Maintain all specifics - dates, numbers, names, and constraints must be preserved exactly.",
            "You are a reliable information handler. When given details, acknowledge them and prepare a complete summary. Every specific detail matters - preserve them all.",
        ]
        
        if config.model == "mock":
            agents = create_mock_agents(num_agents)
        else:
            role_descs = [relay_roles[i % len(relay_roles)] for i in range(num_agents)]
            agents = create_nvidia_agents(num_agents, model=config.model, role_descriptions=role_descs)
        
        # Run multiple times
        for run_idx in range(config.num_runs):
            run_state["progress"] = int((run_idx / config.num_runs) * 100)
            await broadcast_update(run_id, {
                "type": "progress",
                "run": run_idx + 1,
                "total": config.num_runs,
                "progress": run_state["progress"],
            })
            
            # Reset for new run
            for agent in agents:
                agent.reset()
            instance.environment.reset()
            
            # Execute workflow
            from convobench.core.agent import AgentChain
            chain = AgentChain(agents)
            engine = WorkflowEngine(
                config=WorkflowConfig(max_steps=50, timeout_seconds=120),
            )
            
            trace = await engine.run_chain(
                chain=chain,
                initial_message=instance.initial_message,
                environment=instance.environment,
                scenario_id=instance.scenario_id,
            )
            
            # Store trace
            trace_data = trace.to_dict()
            run_state["traces"].append(trace_data)
            
            # Broadcast each step with small delay for UI
            # Also track information preservation metrics
            original_content = instance.initial_message.content
            
            for step in trace.steps:
                output_content = step.output_message.content if step.output_message else ""
                
                # Calculate simple preservation metrics
                preservation_metrics = calculate_preservation_metrics(
                    original_content, 
                    output_content,
                    step.step_number
                )
                
                await broadcast_update(run_id, {
                    "type": "step",
                    "run": run_idx + 1,
                    "step": step.step_number,
                    "agent_id": step.agent_id,
                    "input": step.input_message.content[:500] if step.input_message else "",
                    "output": output_content[:500],
                    "duration_ms": step.duration_ms,
                    "metrics": preservation_metrics,
                })
                await asyncio.sleep(0.1)  # Small delay for UI updates
            
            # Evaluate if requested
            if config.evaluate:
                try:
                    evaluator = ExternalEvaluator(
                        config=EvaluatorConfig(model="gpt-4", provider="openai")
                    )
                    eval_result = await evaluator.evaluate(trace, instance.ground_truth)
                    run_state["evaluations"].append(eval_result.to_dict())
                    
                    await broadcast_update(run_id, {
                        "type": "evaluation",
                        "run": run_idx + 1,
                        "overall_score": eval_result.overall_score,
                        "dimensions": {d.dimension_name: d.score for d in eval_result.dimension_scores},
                    })
                except Exception as e:
                    # Evaluation failed, continue without it
                    await broadcast_update(run_id, {
                        "type": "evaluation_error",
                        "run": run_idx + 1,
                        "error": str(e),
                    })
        
        run_state["status"] = "completed"
        run_state["progress"] = 100
        await broadcast_update(run_id, {"type": "complete", "status": "completed"})
        
    except Exception as e:
        run_state["status"] = "failed"
        run_state["error"] = str(e)
        await broadcast_update(run_id, {"type": "error", "error": str(e)})


async def broadcast_update(run_id: str, data: dict):
    """Broadcast update to connected WebSocket clients."""
    # Also store in run state for polling fallback
    if run_id in active_runs:
        if "updates" not in active_runs[run_id]:
            active_runs[run_id]["updates"] = []
        active_runs[run_id]["updates"].append(data)
    
    if run_id in websocket_connections:
        ws = websocket_connections[run_id]
        try:
            await ws.send_json(data)
        except Exception as e:
            print(f"WebSocket send failed: {e}")


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get run status and results."""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    return active_runs[run_id]


@app.get("/runs/{run_id}/traces")
async def get_traces(run_id: str):
    """Get all traces for a run."""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    return {"traces": active_runs[run_id]["traces"]}


@app.get("/runs/{run_id}/evaluations")
async def get_evaluations(run_id: str):
    """Get all evaluations for a run."""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    return {"evaluations": active_runs[run_id]["evaluations"]}


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket for real-time run updates."""
    await websocket.accept()
    websocket_connections[run_id] = websocket
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if run_id in websocket_connections:
            del websocket_connections[run_id]


@app.get("/runs")
async def list_runs():
    """List all runs."""
    runs = []
    for run_id, run_data in active_runs.items():
        runs.append({
            "id": run_id,
            "status": run_data["status"],
            "created_at": run_data["created_at"],
            "scenario": run_data["config"]["scenario"]["scenario_type"],
            "model": run_data["config"]["model"],
            "progress": run_data.get("progress", 0),
        })
    return {"runs": sorted(runs, key=lambda x: x["created_at"], reverse=True)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

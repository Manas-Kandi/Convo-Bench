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
from convobench.spec import AFVariables, RunManifest, ScenarioPack, ToolConfig, ScenarioRef
from convobench.utils.versioning import get_git_commit_hash
from convobench.store import RunStore
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
    PartialObservabilityReconciliation,
    InterruptDrivenPlanning,
    ProtocolHandoffExperiment,
    FailureModeSuite,
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

# Persistent store + active websocket connections
store = RunStore()
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
    af_variables: Optional[AFVariables] = None
    tool_config: Optional[ToolConfig] = None
    scenario_pack: Optional[ScenarioPack] = None


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
    "partial_observability_reconciliation": lambda cfg: PartialObservabilityReconciliation(
        chain_length=cfg.chain_length,
        domain="incident",
    ),
    "interrupt_driven_planning": lambda cfg: InterruptDrivenPlanning(
        num_agents=cfg.num_agents,
        domain="project",
    ),
    "protocol_handoff_experiment": lambda cfg: ProtocolHandoffExperiment(
        chain_length=cfg.chain_length,
        protocol="structured",
    ),
    "failure_mode_suite": lambda cfg: FailureModeSuite(
        chain_length=cfg.chain_length,
        mode="spec_drift",
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
        {
            "id": "partial_observability_reconciliation",
            "name": "Partial Observability Reconciliation",
            "category": "coordination",
            "description": "Agents see different docs; must reconcile and flag conflicts",
        },
        {
            "id": "interrupt_driven_planning",
            "name": "Interrupt-Driven Planning",
            "category": "planning",
            "description": "Mid-run update forces replanning without losing constraints",
        },
        {
            "id": "protocol_handoff_experiment",
            "name": "Protocol Handoff Experiment",
            "category": "relay",
            "description": "Compare freeform vs structured vs checksum readback handoffs",
        },
        {
            "id": "failure_mode_suite",
            "name": "Failure Mode Suite",
            "category": "adversarial",
            "description": "Stress suite: silent propagation, overconfidence, spec drift",
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
    
    created_at = datetime.utcnow().isoformat()

    # Attach a run manifest for reproducibility
    manifest = RunManifest(
        run_id=run_id,
        code_version=get_git_commit_hash(),
        scenario_pack_version=(config.scenario_pack.version if config.scenario_pack else None),
        af_variables=config.af_variables or (config.scenario_pack.af_variables if config.scenario_pack else AFVariables()),
        tool_config=config.tool_config or ToolConfig(),
        scenarios=(config.scenario_pack.scenarios if config.scenario_pack else [ScenarioRef(scenario_type=config.scenario.scenario_type, params=config.scenario.model_dump())]),
        model_ids=[config.model],
    )
    # Persist run record
    store.upsert_run(
        run_id=run_id,
        created_at=created_at,
        status="pending",
        config_json=config.model_dump(),
        manifest_json=manifest.model_dump(),
    )
    
    # Start run in background
    asyncio.create_task(execute_run(run_id, config))
    
    return RunResponse(run_id=run_id, status="started", message="Run initiated")


async def execute_run(run_id: str, config: RunConfig):
    """Execute a benchmark run."""
    stored = store.get_run(run_id)
    if stored is None:
        return
    store.upsert_run(
        run_id=run_id,
        created_at=stored.created_at,
        status="running",
        config_json=stored.config_json,
        manifest_json=stored.manifest_json,
    )
    
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
            progress_pct = int((run_idx / config.num_runs) * 100)
            await broadcast_update(run_id, {
                "type": "progress",
                "run": run_idx + 1,
                "total": config.num_runs,
                "progress": progress_pct,
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
            store.add_trace(run_id, str(trace.workflow_id), trace.scenario_id, trace_data)

            # Persist deterministic metrics snapshot per workflow
            m = MetricsCollector().collect_from_trace(trace)
            store.add_metrics(run_id, str(trace.workflow_id), trace.scenario_id, m.to_dict())
            
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
                    store.add_evaluation(
                        run_id,
                        str(trace.workflow_id),
                        trace.scenario_id,
                        evaluator.config.model,
                        eval_result.to_dict(),
                    )
                    
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
        
        stored2 = store.get_run(run_id)
        if stored2 is not None:
            store.upsert_run(
                run_id=run_id,
                created_at=stored2.created_at,
                status="completed",
                config_json=stored2.config_json,
                manifest_json=stored2.manifest_json,
            )
        await broadcast_update(run_id, {"type": "complete", "status": "completed"})
        
    except Exception as e:
        stored3 = store.get_run(run_id)
        if stored3 is not None:
            cfg = stored3.config_json
            cfg["error"] = str(e)
            store.upsert_run(
                run_id=run_id,
                created_at=stored3.created_at,
                status="failed",
                config_json=cfg,
                manifest_json=stored3.manifest_json,
            )
        await broadcast_update(run_id, {"type": "error", "error": str(e)})


async def broadcast_update(run_id: str, data: dict):
    """Broadcast update to connected WebSocket clients."""
    # Persist updates into run config_json for polling fallback
    stored = store.get_run(run_id)
    if stored is not None:
        cfg = stored.config_json
        updates = cfg.get("updates", [])
        updates.append(data)
        cfg["updates"] = updates
        store.upsert_run(
            run_id=run_id,
            created_at=stored.created_at,
            status=stored.status,
            config_json=cfg,
            manifest_json=stored.manifest_json,
        )
    
    if run_id in websocket_connections:
        ws = websocket_connections[run_id]
        try:
            await ws.send_json(data)
        except Exception as e:
            print(f"WebSocket send failed: {e}")


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get run status and results."""
    stored = store.get_run(run_id)
    if stored is None:
        raise HTTPException(404, "Run not found")
    return {
        "id": stored.run_id,
        "created_at": stored.created_at,
        "status": stored.status,
        "config": stored.config_json,
        "run_manifest": stored.manifest_json,
        "traces": store.list_traces(run_id),
        "evaluations": store.list_evaluations(run_id),
        "metrics": store.list_metrics(run_id),
    }


@app.get("/runs/{run_id}/traces")
async def get_traces(run_id: str):
    """Get all traces for a run."""
    if store.get_run(run_id) is None:
        raise HTTPException(404, "Run not found")
    return {"traces": store.list_traces(run_id)}


@app.get("/runs/{run_id}/evaluations")
async def get_evaluations(run_id: str):
    """Get all evaluations for a run."""
    if store.get_run(run_id) is None:
        raise HTTPException(404, "Run not found")
    return {"evaluations": store.list_evaluations(run_id)}


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
    runs = store.list_runs(limit=100)
    return {"runs": runs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

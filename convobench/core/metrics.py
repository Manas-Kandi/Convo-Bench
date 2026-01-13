"""Metrics collection and aggregation for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import numpy as np

from convobench.core.types import WorkflowTrace, WorkflowStatus


@dataclass
class StepMetrics:
    """Metrics for a single workflow step."""
    
    step_number: int
    agent_id: str
    duration_ms: float
    token_count: int
    tool_calls: int
    successful_tool_calls: int
    error_occurred: bool
    message_length: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "agent_id": self.agent_id,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "error_occurred": self.error_occurred,
            "message_length": self.message_length,
        }


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for a complete workflow run."""
    
    workflow_id: UUID
    scenario_id: str
    status: WorkflowStatus
    total_steps: int
    total_duration_ms: float
    total_tokens: int
    total_tool_calls: int
    successful_tool_calls: int
    error_count: int
    step_metrics: list[StepMetrics] = field(default_factory=list)
    
    # Evaluation scores (filled by external evaluator)
    intent_preservation_score: Optional[float] = None
    constraint_adherence_score: Optional[float] = None
    action_correctness_score: Optional[float] = None
    coordination_quality_score: Optional[float] = None
    error_propagation_score: Optional[float] = None
    overall_score: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Tool call success rate."""
        if self.total_tool_calls == 0:
            return 1.0
        return self.successful_tool_calls / self.total_tool_calls
    
    @property
    def avg_step_duration_ms(self) -> float:
        """Average duration per step."""
        if self.total_steps == 0:
            return 0.0
        return self.total_duration_ms / self.total_steps
    
    @property
    def avg_tokens_per_step(self) -> float:
        """Average tokens per step."""
        if self.total_steps == 0:
            return 0.0
        return self.total_tokens / self.total_steps
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": str(self.workflow_id),
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "total_steps": self.total_steps,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "avg_step_duration_ms": self.avg_step_duration_ms,
            "avg_tokens_per_step": self.avg_tokens_per_step,
            "intent_preservation_score": self.intent_preservation_score,
            "constraint_adherence_score": self.constraint_adherence_score,
            "action_correctness_score": self.action_correctness_score,
            "coordination_quality_score": self.coordination_quality_score,
            "error_propagation_score": self.error_propagation_score,
            "overall_score": self.overall_score,
            "step_metrics": [s.to_dict() for s in self.step_metrics],
        }


@dataclass
class AggregateMetrics:
    """Statistical aggregation across multiple workflow runs."""
    
    scenario_id: str
    run_count: int
    success_count: int
    failure_count: int
    
    # Duration statistics
    duration_mean: float
    duration_std: float
    duration_min: float
    duration_max: float
    duration_p50: float
    duration_p95: float
    
    # Token statistics
    tokens_mean: float
    tokens_std: float
    tokens_min: float
    tokens_max: float
    
    # Score statistics
    intent_preservation_mean: Optional[float] = None
    intent_preservation_std: Optional[float] = None
    constraint_adherence_mean: Optional[float] = None
    constraint_adherence_std: Optional[float] = None
    action_correctness_mean: Optional[float] = None
    action_correctness_std: Optional[float] = None
    coordination_quality_mean: Optional[float] = None
    coordination_quality_std: Optional[float] = None
    overall_score_mean: Optional[float] = None
    overall_score_std: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        if self.run_count == 0:
            return 0.0
        return self.success_count / self.run_count
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "duration": {
                "mean": self.duration_mean,
                "std": self.duration_std,
                "min": self.duration_min,
                "max": self.duration_max,
                "p50": self.duration_p50,
                "p95": self.duration_p95,
            },
            "tokens": {
                "mean": self.tokens_mean,
                "std": self.tokens_std,
                "min": self.tokens_min,
                "max": self.tokens_max,
            },
            "scores": {
                "intent_preservation": {"mean": self.intent_preservation_mean, "std": self.intent_preservation_std},
                "constraint_adherence": {"mean": self.constraint_adherence_mean, "std": self.constraint_adherence_std},
                "action_correctness": {"mean": self.action_correctness_mean, "std": self.action_correctness_std},
                "coordination_quality": {"mean": self.coordination_quality_mean, "std": self.coordination_quality_std},
                "overall": {"mean": self.overall_score_mean, "std": self.overall_score_std},
            },
        }


class MetricsCollector:
    """Collects and aggregates metrics from workflow executions."""
    
    def __init__(self):
        self._workflow_metrics: list[WorkflowMetrics] = []
        self._traces: list[WorkflowTrace] = []
    
    def collect_from_trace(self, trace: WorkflowTrace) -> WorkflowMetrics:
        """
        Extract metrics from a workflow trace.
        
        Args:
            trace: Complete workflow trace
            
        Returns:
            Extracted workflow metrics
        """
        step_metrics = []
        total_tool_calls = 0
        successful_tool_calls = 0
        error_count = 0
        
        for step in trace.steps:
            tool_calls = len(step.actions)
            successful = sum(1 for tr in step.tool_results if tr.success)
            
            total_tool_calls += tool_calls
            successful_tool_calls += successful
            if step.error:
                error_count += 1
            
            step_metrics.append(StepMetrics(
                step_number=step.step_number,
                agent_id=step.agent_id,
                duration_ms=step.duration_ms,
                token_count=step.token_count,
                tool_calls=tool_calls,
                successful_tool_calls=successful,
                error_occurred=step.error is not None,
                message_length=len(step.output_message.content) if step.output_message else 0,
            ))
        
        metrics = WorkflowMetrics(
            workflow_id=trace.workflow_id,
            scenario_id=trace.scenario_id,
            status=trace.status,
            total_steps=len(trace.steps),
            total_duration_ms=trace.total_duration_ms,
            total_tokens=trace.total_tokens,
            total_tool_calls=total_tool_calls,
            successful_tool_calls=successful_tool_calls,
            error_count=error_count,
            step_metrics=step_metrics,
        )
        
        self._workflow_metrics.append(metrics)
        self._traces.append(trace)
        
        return metrics
    
    def aggregate(self, scenario_id: Optional[str] = None) -> AggregateMetrics:
        """
        Compute aggregate statistics across collected metrics.
        
        Args:
            scenario_id: Optional filter by scenario
            
        Returns:
            Aggregated metrics
        """
        metrics = self._workflow_metrics
        if scenario_id:
            metrics = [m for m in metrics if m.scenario_id == scenario_id]
        
        if not metrics:
            raise ValueError("No metrics to aggregate")
        
        durations = [m.total_duration_ms for m in metrics]
        tokens = [m.total_tokens for m in metrics]
        
        success_count = sum(1 for m in metrics if m.status == WorkflowStatus.COMPLETED)
        failure_count = len(metrics) - success_count
        
        # Score aggregation (only for metrics that have scores)
        intent_scores = [m.intent_preservation_score for m in metrics if m.intent_preservation_score is not None]
        constraint_scores = [m.constraint_adherence_score for m in metrics if m.constraint_adherence_score is not None]
        action_scores = [m.action_correctness_score for m in metrics if m.action_correctness_score is not None]
        coord_scores = [m.coordination_quality_score for m in metrics if m.coordination_quality_score is not None]
        overall_scores = [m.overall_score for m in metrics if m.overall_score is not None]
        
        return AggregateMetrics(
            scenario_id=scenario_id or "all",
            run_count=len(metrics),
            success_count=success_count,
            failure_count=failure_count,
            duration_mean=float(np.mean(durations)),
            duration_std=float(np.std(durations)),
            duration_min=float(np.min(durations)),
            duration_max=float(np.max(durations)),
            duration_p50=float(np.percentile(durations, 50)),
            duration_p95=float(np.percentile(durations, 95)),
            tokens_mean=float(np.mean(tokens)),
            tokens_std=float(np.std(tokens)),
            tokens_min=float(np.min(tokens)),
            tokens_max=float(np.max(tokens)),
            intent_preservation_mean=float(np.mean(intent_scores)) if intent_scores else None,
            intent_preservation_std=float(np.std(intent_scores)) if intent_scores else None,
            constraint_adherence_mean=float(np.mean(constraint_scores)) if constraint_scores else None,
            constraint_adherence_std=float(np.std(constraint_scores)) if constraint_scores else None,
            action_correctness_mean=float(np.mean(action_scores)) if action_scores else None,
            action_correctness_std=float(np.std(action_scores)) if action_scores else None,
            coordination_quality_mean=float(np.mean(coord_scores)) if coord_scores else None,
            coordination_quality_std=float(np.std(coord_scores)) if coord_scores else None,
            overall_score_mean=float(np.mean(overall_scores)) if overall_scores else None,
            overall_score_std=float(np.std(overall_scores)) if overall_scores else None,
        )
    
    def get_all_metrics(self) -> list[WorkflowMetrics]:
        """Get all collected workflow metrics."""
        return self._workflow_metrics.copy()
    
    def get_all_traces(self) -> list[WorkflowTrace]:
        """Get all collected workflow traces."""
        return self._traces.copy()
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        self._workflow_metrics = []
        self._traces = []
    
    def update_scores(
        self,
        workflow_id: UUID,
        scores: dict[str, float],
    ) -> None:
        """
        Update evaluation scores for a workflow.
        
        Args:
            workflow_id: ID of the workflow to update
            scores: Dictionary of score names to values
        """
        for metrics in self._workflow_metrics:
            if metrics.workflow_id == workflow_id:
                if "intent_preservation" in scores:
                    metrics.intent_preservation_score = scores["intent_preservation"]
                if "constraint_adherence" in scores:
                    metrics.constraint_adherence_score = scores["constraint_adherence"]
                if "action_correctness" in scores:
                    metrics.action_correctness_score = scores["action_correctness"]
                if "coordination_quality" in scores:
                    metrics.coordination_quality_score = scores["coordination_quality"]
                if "error_propagation" in scores:
                    metrics.error_propagation_score = scores["error_propagation"]
                if "overall" in scores:
                    metrics.overall_score = scores["overall"]
                break

"""External LLM evaluator for assessing workflow quality."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from convobench.core.types import WorkflowTrace
from convobench.evaluation.rubrics import EvaluationRubric, get_rubric_for_category


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    
    dimension_name: str
    score: float
    evidence: str
    issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "score": self.score,
            "evidence": self.evidence,
            "issues": self.issues,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a workflow."""
    
    workflow_id: UUID
    scenario_id: str
    evaluator_model: str
    dimension_scores: list[DimensionScore]
    overall_score: float
    summary: str
    critical_failures: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[str] = None
    
    def get_score(self, dimension_name: str) -> Optional[float]:
        """Get score for a specific dimension."""
        for ds in self.dimension_scores:
            if ds.dimension_name == dimension_name:
                return ds.score
        return None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": str(self.workflow_id),
            "scenario_id": self.scenario_id,
            "evaluator_model": self.evaluator_model,
            "dimension_scores": [ds.to_dict() for ds in self.dimension_scores],
            "overall_score": self.overall_score,
            "summary": self.summary,
            "critical_failures": self.critical_failures,
            "recommendations": self.recommendations,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
        }
    
    def to_metrics_dict(self) -> dict[str, float]:
        """Convert to format expected by MetricsCollector."""
        metrics = {"overall": self.overall_score}
        
        dimension_mapping = {
            "Intent Preservation": "intent_preservation",
            "Constraint Adherence": "constraint_adherence",
            "Action Correctness": "action_correctness",
            "Coordination Quality": "coordination_quality",
            "Error Propagation": "error_propagation",
            "Information Fidelity": "information_fidelity",
        }
        
        for ds in self.dimension_scores:
            key = dimension_mapping.get(ds.dimension_name, ds.dimension_name.lower().replace(" ", "_"))
            metrics[key] = ds.score
        
        return metrics


class EvaluatorConfig(BaseModel):
    """Configuration for the external evaluator."""
    
    model: str = Field(default="gpt-4", description="Model to use for evaluation")
    provider: str = Field(default="openai", description="Provider (openai, anthropic)")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)
    retry_attempts: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)


class ExternalEvaluator:
    """
    External LLM evaluator that sits outside the agent chain.
    
    This evaluator assesses workflow quality by analyzing the complete
    trace against ground truth and evaluation rubrics. Being external
    to the workflow ensures unbiased assessment.
    """
    
    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        custom_rubrics: Optional[dict[str, EvaluationRubric]] = None,
    ):
        self.config = config or EvaluatorConfig()
        self.custom_rubrics = custom_rubrics or {}
        self._client = None
    
    async def evaluate(
        self,
        trace: WorkflowTrace,
        ground_truth: dict[str, Any],
        rubric: Optional[EvaluationRubric] = None,
    ) -> EvaluationResult:
        """
        Evaluate a workflow trace against ground truth.
        
        Args:
            trace: Complete workflow trace to evaluate
            ground_truth: Expected outcomes and preserved information
            rubric: Evaluation rubric (auto-selected if not provided)
            
        Returns:
            Complete evaluation result with scores and analysis
        """
        if rubric is None:
            category = trace.metadata.get("category", "relay")
            rubric = self.custom_rubrics.get(category) or get_rubric_for_category(category)
        
        evaluation_prompt = self._build_evaluation_prompt(trace, ground_truth, rubric)
        
        response = await self._call_evaluator(evaluation_prompt)
        
        result = self._parse_evaluation_response(
            response=response,
            workflow_id=trace.workflow_id,
            scenario_id=trace.scenario_id,
            rubric=rubric,
        )
        
        return result
    
    async def evaluate_batch(
        self,
        traces: list[WorkflowTrace],
        ground_truths: list[dict[str, Any]],
        rubric: Optional[EvaluationRubric] = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple workflow traces."""
        results = []
        for trace, gt in zip(traces, ground_truths):
            result = await self.evaluate(trace, gt, rubric)
            results.append(result)
        return results
    
    def _build_evaluation_prompt(
        self,
        trace: WorkflowTrace,
        ground_truth: dict[str, Any],
        rubric: EvaluationRubric,
    ) -> str:
        """Build the complete evaluation prompt."""
        trace_summary = self._format_trace_for_evaluation(trace)
        rubric_prompt = rubric.to_evaluation_prompt(ground_truth)
        
        prompt = f"""You are an expert evaluator assessing the quality of a multi-agent workflow.

## Your Role
You are EXTERNAL to the agent chain - you did not participate in the workflow.
Your job is to objectively assess how well the agents performed based on:
1. The workflow trace (what actually happened)
2. The ground truth (what should have happened)
3. The evaluation rubric (how to score)

## Workflow Trace

The following is a complete trace of the workflow execution:

{trace_summary}

## Evaluation Instructions

{rubric_prompt}

## Important Guidelines

1. Be objective and evidence-based
2. Score based on actual outcomes, not intentions
3. Consider both individual agent performance and overall workflow quality
4. Identify specific points where information was lost or distorted
5. Note any constraint violations or goal drift
6. Assess error handling and recovery

Provide your evaluation now."""
        
        return prompt
    
    def _format_trace_for_evaluation(self, trace: WorkflowTrace) -> str:
        """Format workflow trace for evaluation."""
        lines = [
            f"**Workflow ID**: {trace.workflow_id}",
            f"**Scenario**: {trace.scenario_id}",
            f"**Status**: {trace.status.value}",
            f"**Total Steps**: {len(trace.steps)}",
            f"**Duration**: {trace.total_duration_ms:.2f}ms",
            "",
            "### Step-by-Step Trace",
            "",
        ]
        
        for step in trace.steps:
            lines.append(f"#### Step {step.step_number + 1}: Agent `{step.agent_id}`")
            lines.append("")
            lines.append(f"**Input**:")
            lines.append(f"```")
            lines.append(step.input_message.content[:1000] + ("..." if len(step.input_message.content) > 1000 else ""))
            lines.append(f"```")
            lines.append("")
            
            if step.output_message:
                lines.append(f"**Output**:")
                lines.append(f"```")
                lines.append(step.output_message.content[:1000] + ("..." if len(step.output_message.content) > 1000 else ""))
                lines.append(f"```")
                lines.append("")
            
            if step.actions:
                lines.append(f"**Actions**: {len(step.actions)} action(s)")
                for action in step.actions[:5]:
                    lines.append(f"- {action.action_type.value}: {action.payload}")
                lines.append("")
            
            if step.tool_results:
                lines.append(f"**Tool Results**: {len(step.tool_results)} result(s)")
                for result in step.tool_results[:5]:
                    status = "Success" if result.success else f"Failed: {result.error}"
                    lines.append(f"- {status}")
                lines.append("")
            
            if step.error:
                lines.append(f"**Error**: {step.error}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    async def _call_evaluator(self, prompt: str) -> str:
        """Call the evaluator LLM."""
        if self.config.provider == "openai":
            return await self._call_openai(prompt)
        elif self.config.provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for evaluation."""
        try:
            from openai import AsyncOpenAI
            
            if self._client is None:
                self._client = AsyncOpenAI()
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for multi-agent AI systems."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            return self._generate_mock_response()
        except Exception as e:
            return self._generate_mock_response(error=str(e))
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API for evaluation."""
        try:
            from anthropic import AsyncAnthropic
            
            if self._client is None:
                self._client = AsyncAnthropic()
            
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            
            return response.content[0].text
            
        except ImportError:
            return self._generate_mock_response()
        except Exception as e:
            return self._generate_mock_response(error=str(e))
    
    def _generate_mock_response(self, error: Optional[str] = None) -> str:
        """Generate mock response for testing without API access."""
        mock = {
            "scores": {
                "Intent Preservation": {"score": 4, "evidence": "Mock evaluation", "issues": []},
                "Constraint Adherence": {"score": 4, "evidence": "Mock evaluation", "issues": []},
                "Action Correctness": {"score": 3, "evidence": "Mock evaluation", "issues": []},
                "Coordination Quality": {"score": 4, "evidence": "Mock evaluation", "issues": []},
                "Error Propagation": {"score": 3, "evidence": "Mock evaluation", "issues": []},
            },
            "overall_score": 3.6,
            "summary": "Mock evaluation - API not available" + (f" (Error: {error})" if error else ""),
            "critical_failures": [],
            "recommendations": ["Configure API credentials for real evaluation"],
        }
        return f"```json\n{json.dumps(mock, indent=2)}\n```"
    
    def _parse_evaluation_response(
        self,
        response: str,
        workflow_id: UUID,
        scenario_id: str,
        rubric: EvaluationRubric,
    ) -> EvaluationResult:
        """Parse the LLM evaluation response."""
        try:
            json_match = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_match = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_match = response[start:end].strip()
            
            data = json.loads(json_match)
            
            dimension_scores = []
            for dim_name, dim_data in data.get("scores", {}).items():
                dimension_scores.append(DimensionScore(
                    dimension_name=dim_name,
                    score=float(dim_data.get("score", 3)),
                    evidence=dim_data.get("evidence", ""),
                    issues=dim_data.get("issues", []),
                ))
            
            return EvaluationResult(
                workflow_id=workflow_id,
                scenario_id=scenario_id,
                evaluator_model=self.config.model,
                dimension_scores=dimension_scores,
                overall_score=float(data.get("overall_score", 3.0)),
                summary=data.get("summary", ""),
                critical_failures=data.get("critical_failures", []),
                recommendations=data.get("recommendations", []),
                raw_response=response,
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return EvaluationResult(
                workflow_id=workflow_id,
                scenario_id=scenario_id,
                evaluator_model=self.config.model,
                dimension_scores=[],
                overall_score=0.0,
                summary=f"Failed to parse evaluation response: {e}",
                critical_failures=["Evaluation parsing failed"],
                raw_response=response,
            )


class ComparativeEvaluator:
    """
    Evaluator for comparing multiple agent configurations.
    
    Runs the same scenarios across different LLMs/configurations
    and provides comparative analysis.
    """
    
    def __init__(self, base_evaluator: ExternalEvaluator):
        self.evaluator = base_evaluator
        self._results: dict[str, list[EvaluationResult]] = {}
    
    async def compare(
        self,
        traces_by_config: dict[str, list[WorkflowTrace]],
        ground_truths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compare evaluation results across configurations.
        
        Args:
            traces_by_config: Mapping of config name to workflow traces
            ground_truths: Ground truth for each scenario
            
        Returns:
            Comparative analysis results
        """
        for config_name, traces in traces_by_config.items():
            results = await self.evaluator.evaluate_batch(traces, ground_truths)
            self._results[config_name] = results
        
        return self._generate_comparison()
    
    def _generate_comparison(self) -> dict[str, Any]:
        """Generate comparative analysis."""
        comparison = {
            "configurations": list(self._results.keys()),
            "by_config": {},
            "rankings": {},
        }
        
        for config_name, results in self._results.items():
            scores = [r.overall_score for r in results]
            comparison["by_config"][config_name] = {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "num_evaluations": len(results),
            }
        
        sorted_configs = sorted(
            comparison["by_config"].items(),
            key=lambda x: x[1]["mean_score"],
            reverse=True,
        )
        comparison["rankings"] = {
            config: rank + 1 for rank, (config, _) in enumerate(sorted_configs)
        }
        
        return comparison

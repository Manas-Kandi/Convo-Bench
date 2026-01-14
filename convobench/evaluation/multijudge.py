"""Multi-judge evaluation robustness utilities.

Implements:
- MultiJudgeEvaluator: runs multiple ExternalEvaluator instances
- Aggregation: mean/median overall score, per-dimension aggregation
- Inter-rater reliability: Krippendorff's alpha (interval) fallback to simple agreement
- Calibration tasks: lightweight schema to store judge calibration cases

Design goal: keep core benchmark runnable without additional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from convobench.core.types import WorkflowTrace
from convobench.evaluation.evaluator import EvaluationResult, ExternalEvaluator


@dataclass
class JudgeCalibrationTask:
    """A calibration task with an expected score pattern."""

    task_id: str
    prompt: str
    expected_overall_range: tuple[float, float] = (0.0, 5.0)


@dataclass
class MultiJudgeResult:
    workflow_id: str
    scenario_id: str
    judge_results: List[EvaluationResult]
    overall_mean: float
    overall_median: float
    inter_rater_alpha: Optional[float]

    # aggregated by dimension name
    dimension_means: Dict[str, float]


def krippendorff_alpha_interval(matrix: List[List[float]]) -> Optional[float]:
    """Compute Krippendorff's alpha for interval data.

    matrix: rows=items, cols=judges, values=score or None.

    Returns None if insufficient data.
    """
    # Remove items with <2 ratings
    items = []
    for row in matrix:
        vals = [v for v in row if v is not None]
        if len(vals) >= 2:
            items.append(row)

    if len(items) < 2:
        return None

    # Observed disagreement Do
    Do = 0.0
    n_pairs = 0
    for row in items:
        vals = [v for v in row if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                Do += (vals[i] - vals[j]) ** 2
                n_pairs += 1
    if n_pairs == 0:
        return None
    Do /= n_pairs

    # Expected disagreement De based on pooled distribution
    pooled = [v for row in items for v in row if v is not None]
    if len(pooled) < 2:
        return None

    mu = float(np.mean(pooled))
    De = float(np.mean([(v - mu) ** 2 for v in pooled])) * 2

    if De == 0:
        return 1.0

    return 1.0 - (Do / De)


class MultiJudgeEvaluator:
    def __init__(self, judges: List[ExternalEvaluator]):
        if len(judges) < 1:
            raise ValueError("MultiJudgeEvaluator requires at least one judge")
        self.judges = judges

    async def evaluate(
        self,
        trace: WorkflowTrace,
        ground_truth: dict[str, Any],
    ) -> MultiJudgeResult:
        results: List[EvaluationResult] = []
        for j in self.judges:
            results.append(await j.evaluate(trace, ground_truth))

        overall_scores = [r.overall_score for r in results]
        dim_scores: Dict[str, List[float]] = {}
        for r in results:
            for ds in r.dimension_scores:
                dim_scores.setdefault(ds.dimension_name, []).append(ds.score)

        dim_means = {k: float(mean(v)) for k, v in dim_scores.items() if v}

        # Inter-rater alpha on overall scores (single item) is not meaningful; compute on per-dimension matrix.
        # We'll compute alpha on a matrix of items=dimensions, judges=scores.
        dims = sorted(dim_scores.keys())
        if dims:
            matrix: List[List[float]] = []
            for d in dims:
                row = []
                # for each judge, find their score for dim (or None)
                for r in results:
                    s = None
                    for ds in r.dimension_scores:
                        if ds.dimension_name == d:
                            s = float(ds.score)
                            break
                    row.append(s)
                matrix.append(row)
            alpha = krippendorff_alpha_interval(matrix)
        else:
            alpha = None

        return MultiJudgeResult(
            workflow_id=str(trace.workflow_id),
            scenario_id=trace.scenario_id,
            judge_results=results,
            overall_mean=float(mean(overall_scores)) if overall_scores else 0.0,
            overall_median=float(median(overall_scores)) if overall_scores else 0.0,
            inter_rater_alpha=alpha,
            dimension_means=dim_means,
        )

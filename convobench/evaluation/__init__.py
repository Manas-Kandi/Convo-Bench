"""Evaluation framework for ConvoBench."""

from convobench.evaluation.evaluator import ExternalEvaluator, EvaluationResult
from convobench.evaluation.rubrics import EvaluationRubric, RubricDimension
from convobench.evaluation.analysis import StatisticalAnalyzer, ComparisonReport

__all__ = [
    "ExternalEvaluator",
    "EvaluationResult",
    "EvaluationRubric",
    "RubricDimension",
    "StatisticalAnalyzer",
    "ComparisonReport",
]

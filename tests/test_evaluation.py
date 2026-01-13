"""Tests for evaluation framework."""

import pytest
from uuid import uuid4

from convobench.core.types import (
    Message,
    MessageRole,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from convobench.evaluation.analysis import StatisticalAnalyzer, generate_leaderboard
from convobench.evaluation.evaluator import DimensionScore, EvaluationResult, ExternalEvaluator
from convobench.evaluation.rubrics import (
    EvaluationRubric,
    RubricDimension,
    ScoreLevel,
    create_relay_rubric,
    create_planning_rubric,
    get_rubric_for_category,
)


class TestRubricDimension:
    """Tests for RubricDimension."""
    
    def test_default_score_descriptions(self):
        dim = RubricDimension(
            name="Test Dimension",
            description="A test dimension",
        )
        
        assert ScoreLevel.EXCELLENT in dim.score_descriptions
        assert ScoreLevel.FAILING in dim.score_descriptions
    
    def test_custom_score_descriptions(self):
        dim = RubricDimension(
            name="Custom",
            description="Custom dimension",
            score_descriptions={
                ScoreLevel.EXCELLENT: "Perfect",
                ScoreLevel.GOOD: "Good",
                ScoreLevel.ACCEPTABLE: "OK",
                ScoreLevel.POOR: "Bad",
                ScoreLevel.FAILING: "Failed",
            },
        )
        
        assert dim.score_descriptions[ScoreLevel.EXCELLENT] == "Perfect"
    
    def test_to_prompt_section(self):
        dim = RubricDimension(
            name="Test",
            description="Test description",
            weight=1.5,
        )
        
        prompt = dim.to_prompt_section()
        assert "### Test" in prompt
        assert "Test description" in prompt
        assert "1.5" in prompt


class TestEvaluationRubric:
    """Tests for EvaluationRubric."""
    
    def test_get_total_weight(self):
        rubric = EvaluationRubric(
            name="Test",
            description="Test rubric",
            dimensions=[
                RubricDimension(name="D1", description="", weight=1.0),
                RubricDimension(name="D2", description="", weight=2.0),
                RubricDimension(name="D3", description="", weight=1.5),
            ],
        )
        
        assert rubric.get_total_weight() == 4.5
    
    def test_normalize_weights(self):
        rubric = EvaluationRubric(
            name="Test",
            description="Test rubric",
            dimensions=[
                RubricDimension(name="D1", description="", weight=2.0),
                RubricDimension(name="D2", description="", weight=2.0),
            ],
        )
        
        rubric.normalize_weights()
        
        assert rubric.dimensions[0].weight == 0.5
        assert rubric.dimensions[1].weight == 0.5
    
    def test_to_evaluation_prompt(self):
        rubric = create_relay_rubric()
        ground_truth = {"expected": "test"}
        
        prompt = rubric.to_evaluation_prompt(ground_truth)
        
        assert "Information Relay Evaluation" in prompt
        assert "Intent Preservation" in prompt
        assert "expected" in prompt


class TestRubricFactories:
    """Tests for rubric factory functions."""
    
    def test_create_relay_rubric(self):
        rubric = create_relay_rubric()
        
        assert rubric.name == "Information Relay Evaluation"
        assert len(rubric.dimensions) > 0
        
        dim_names = [d.name for d in rubric.dimensions]
        assert "Information Fidelity" in dim_names
    
    def test_create_planning_rubric(self):
        rubric = create_planning_rubric()
        
        assert rubric.name == "Collaborative Planning Evaluation"
        dim_names = [d.name for d in rubric.dimensions]
        assert "Plan Completeness" in dim_names
    
    def test_get_rubric_for_category(self):
        for category in ["relay", "planning", "coordination", "adversarial"]:
            rubric = get_rubric_for_category(category)
            assert rubric is not None
            assert len(rubric.dimensions) > 0


class TestEvaluationResult:
    """Tests for EvaluationResult."""
    
    def test_get_score(self):
        result = EvaluationResult(
            workflow_id=uuid4(),
            scenario_id="test",
            evaluator_model="gpt-4",
            dimension_scores=[
                DimensionScore(dimension_name="Intent", score=4.5, evidence="Good"),
                DimensionScore(dimension_name="Action", score=3.0, evidence="OK"),
            ],
            overall_score=3.75,
            summary="Test summary",
        )
        
        assert result.get_score("Intent") == 4.5
        assert result.get_score("Action") == 3.0
        assert result.get_score("Nonexistent") is None
    
    def test_to_metrics_dict(self):
        result = EvaluationResult(
            workflow_id=uuid4(),
            scenario_id="test",
            evaluator_model="gpt-4",
            dimension_scores=[
                DimensionScore(dimension_name="Intent Preservation", score=4.0, evidence=""),
            ],
            overall_score=4.0,
            summary="",
        )
        
        metrics = result.to_metrics_dict()
        
        assert metrics["overall"] == 4.0
        assert "intent_preservation" in metrics


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer."""
    
    def test_summarize_basic(self):
        analyzer = StatisticalAnalyzer()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        summary = analyzer.summarize(values)
        
        assert summary.mean == 3.0
        assert summary.median == 3.0
        assert summary.min_val == 1.0
        assert summary.max_val == 5.0
        assert summary.n == 5
    
    def test_summarize_empty(self):
        analyzer = StatisticalAnalyzer()
        summary = analyzer.summarize([])
        
        assert summary.mean == 0
        assert summary.n == 0
    
    def test_compare_two_significant(self):
        analyzer = StatisticalAnalyzer()
        
        values_a = [4.0, 4.5, 4.2, 4.3, 4.1]
        values_b = [2.0, 2.5, 2.2, 2.3, 2.1]
        
        result = analyzer.compare_two(values_a, values_b, "config_a", "config_b", "score")
        
        assert result.mean_diff > 0
        assert result.significant is True
        assert result.winner == "config_a"
    
    def test_compare_two_not_significant(self):
        analyzer = StatisticalAnalyzer()
        
        values_a = [3.0, 3.1, 2.9, 3.0, 3.1]
        values_b = [3.0, 2.9, 3.1, 3.0, 2.9]
        
        result = analyzer.compare_two(values_a, values_b, "config_a", "config_b", "score")
        
        assert result.significant is False
        assert result.winner is None
    
    def test_compute_reliability(self):
        analyzer = StatisticalAnalyzer()
        
        results = [
            EvaluationResult(
                workflow_id=uuid4(),
                scenario_id="test",
                evaluator_model="gpt-4",
                dimension_scores=[],
                overall_score=score,
                summary="",
            )
            for score in [4.0, 4.1, 3.9, 4.0, 4.2, 3.8, 4.0, 4.1, 3.9, 4.0]
        ]
        
        reliability = analyzer.compute_reliability(results)
        
        assert "coefficient_of_variation" in reliability
        assert "reliability_rating" in reliability


class TestLeaderboard:
    """Tests for leaderboard generation."""
    
    def test_generate_leaderboard(self):
        results_by_config = {
            "config_a": [
                EvaluationResult(
                    workflow_id=uuid4(),
                    scenario_id="test",
                    evaluator_model="gpt-4",
                    dimension_scores=[],
                    overall_score=4.0,
                    summary="",
                )
                for _ in range(5)
            ],
            "config_b": [
                EvaluationResult(
                    workflow_id=uuid4(),
                    scenario_id="test",
                    evaluator_model="gpt-4",
                    dimension_scores=[],
                    overall_score=3.0,
                    summary="",
                )
                for _ in range(5)
            ],
        }
        
        leaderboard = generate_leaderboard(results_by_config)
        
        assert len(leaderboard) == 2
        assert leaderboard[0]["configuration"] == "config_a"
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[1]["configuration"] == "config_b"
        assert leaderboard[1]["rank"] == 2

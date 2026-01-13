"""Statistical analysis and reporting for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats

from convobench.core.metrics import AggregateMetrics, WorkflowMetrics
from convobench.evaluation.evaluator import EvaluationResult


@dataclass
class StatisticalSummary:
    """Statistical summary for a set of scores."""
    
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    p25: float
    p75: float
    p95: float
    n: int
    ci_lower: float  # 95% confidence interval
    ci_upper: float
    
    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min_val,
            "max": self.max_val,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "n": self.n,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two configurations."""
    
    config_a: str
    config_b: str
    metric: str
    mean_diff: float
    effect_size: float  # Cohen's d
    p_value: float
    significant: bool
    winner: Optional[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "config_a": self.config_a,
            "config_b": self.config_b,
            "metric": self.metric,
            "mean_difference": self.mean_diff,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "statistically_significant": self.significant,
            "winner": self.winner,
        }


@dataclass
class ComparisonReport:
    """Complete comparison report across configurations."""
    
    configurations: list[str]
    scenario_id: str
    metric_summaries: dict[str, dict[str, StatisticalSummary]]
    pairwise_comparisons: list[ComparisonResult]
    rankings: dict[str, dict[str, int]]
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "configurations": self.configurations,
            "scenario_id": self.scenario_id,
            "metric_summaries": {
                metric: {config: summary.to_dict() for config, summary in configs.items()}
                for metric, configs in self.metric_summaries.items()
            },
            "pairwise_comparisons": [c.to_dict() for c in self.pairwise_comparisons],
            "rankings": self.rankings,
            "recommendations": self.recommendations,
        }
    
    def get_best_config(self, metric: str = "overall_score") -> Optional[str]:
        """Get the best performing configuration for a metric."""
        if metric in self.rankings:
            for config, rank in self.rankings[metric].items():
                if rank == 1:
                    return config
        return None


class StatisticalAnalyzer:
    """
    Statistical analyzer for benchmark results.
    
    Provides tools for analyzing individual configurations,
    comparing across configurations, and generating reports.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def summarize(self, values: list[float]) -> StatisticalSummary:
        """Compute statistical summary for a list of values."""
        if not values:
            return StatisticalSummary(
                mean=0, std=0, median=0, min_val=0, max_val=0,
                p25=0, p75=0, p95=0, n=0, ci_lower=0, ci_upper=0,
            )
        
        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0
        
        # 95% confidence interval
        if n > 1 and std > 0:
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean
        
        return StatisticalSummary(
            mean=mean,
            std=std,
            median=float(np.median(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            n=n,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
    
    def compare_two(
        self,
        values_a: list[float],
        values_b: list[float],
        config_a: str,
        config_b: str,
        metric: str,
    ) -> ComparisonResult:
        """
        Compare two configurations using statistical tests.
        
        Uses Welch's t-test for comparing means and Cohen's d for effect size.
        """
        if not values_a or not values_b:
            return ComparisonResult(
                config_a=config_a,
                config_b=config_b,
                metric=metric,
                mean_diff=0,
                effect_size=0,
                p_value=1.0,
                significant=False,
                winner=None,
            )
        
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)
        
        mean_a = np.mean(arr_a)
        mean_b = np.mean(arr_b)
        mean_diff = float(mean_a - mean_b)
        
        # Welch's t-test (does not assume equal variances)
        t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(arr_a) - 1) * np.var(arr_a, ddof=1) + 
             (len(arr_b) - 1) * np.var(arr_b, ddof=1)) /
            (len(arr_a) + len(arr_b) - 2)
        )
        effect_size = float(mean_diff / pooled_std) if pooled_std > 0 else 0
        
        significant = p_value < self.significance_level
        
        if significant:
            winner = config_a if mean_diff > 0 else config_b
        else:
            winner = None
        
        return ComparisonResult(
            config_a=config_a,
            config_b=config_b,
            metric=metric,
            mean_diff=mean_diff,
            effect_size=effect_size,
            p_value=float(p_value),
            significant=significant,
            winner=winner,
        )
    
    def analyze_results(
        self,
        results_by_config: dict[str, list[EvaluationResult]],
        scenario_id: str = "all",
    ) -> ComparisonReport:
        """
        Analyze and compare results across multiple configurations.
        
        Args:
            results_by_config: Mapping of config name to evaluation results
            scenario_id: Scenario identifier for the report
            
        Returns:
            Complete comparison report
        """
        configurations = list(results_by_config.keys())
        
        # Extract scores by metric
        metrics = ["overall_score", "intent_preservation", "constraint_adherence",
                   "action_correctness", "coordination_quality", "error_propagation"]
        
        scores_by_metric: dict[str, dict[str, list[float]]] = {
            metric: {} for metric in metrics
        }
        
        for config, results in results_by_config.items():
            for metric in metrics:
                scores = []
                for r in results:
                    if metric == "overall_score":
                        scores.append(r.overall_score)
                    else:
                        score = r.get_score(metric.replace("_", " ").title())
                        if score is not None:
                            scores.append(score)
                scores_by_metric[metric][config] = scores
        
        # Compute summaries
        metric_summaries: dict[str, dict[str, StatisticalSummary]] = {}
        for metric, config_scores in scores_by_metric.items():
            metric_summaries[metric] = {
                config: self.summarize(scores)
                for config, scores in config_scores.items()
            }
        
        # Pairwise comparisons
        pairwise_comparisons = []
        for i, config_a in enumerate(configurations):
            for config_b in configurations[i + 1:]:
                for metric in metrics:
                    comparison = self.compare_two(
                        scores_by_metric[metric].get(config_a, []),
                        scores_by_metric[metric].get(config_b, []),
                        config_a,
                        config_b,
                        metric,
                    )
                    pairwise_comparisons.append(comparison)
        
        # Rankings
        rankings: dict[str, dict[str, int]] = {}
        for metric in metrics:
            sorted_configs = sorted(
                configurations,
                key=lambda c: metric_summaries[metric][c].mean,
                reverse=True,
            )
            rankings[metric] = {config: rank + 1 for rank, config in enumerate(sorted_configs)}
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            configurations, metric_summaries, pairwise_comparisons
        )
        
        return ComparisonReport(
            configurations=configurations,
            scenario_id=scenario_id,
            metric_summaries=metric_summaries,
            pairwise_comparisons=pairwise_comparisons,
            rankings=rankings,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        configurations: list[str],
        summaries: dict[str, dict[str, StatisticalSummary]],
        comparisons: list[ComparisonResult],
    ) -> list[str]:
        """Generate actionable recommendations from analysis."""
        recommendations = []
        
        # Find overall best performer
        overall_scores = {
            config: summaries["overall_score"][config].mean
            for config in configurations
        }
        best_config = max(overall_scores, key=overall_scores.get)
        recommendations.append(
            f"Best overall performer: {best_config} "
            f"(mean score: {overall_scores[best_config]:.2f})"
        )
        
        # Identify significant differences
        significant_diffs = [c for c in comparisons if c.significant]
        if significant_diffs:
            recommendations.append(
                f"Found {len(significant_diffs)} statistically significant differences"
            )
        
        # Identify weak areas for each config
        for config in configurations:
            weak_metrics = []
            for metric, config_summaries in summaries.items():
                if metric == "overall_score":
                    continue
                if config_summaries[config].mean < 3.0:
                    weak_metrics.append(metric)
            
            if weak_metrics:
                recommendations.append(
                    f"{config} shows weakness in: {', '.join(weak_metrics)}"
                )
        
        return recommendations
    
    def compute_reliability(
        self,
        results: list[EvaluationResult],
        min_runs: int = 10,
    ) -> dict[str, float]:
        """
        Compute reliability metrics for benchmark results.
        
        Returns coefficient of variation and other reliability indicators.
        """
        if len(results) < min_runs:
            return {"warning": f"Insufficient runs ({len(results)} < {min_runs})"}
        
        scores = [r.overall_score for r in results]
        summary = self.summarize(scores)
        
        # Coefficient of variation (lower is more reliable)
        cv = summary.std / summary.mean if summary.mean > 0 else float('inf')
        
        # Interquartile range relative to median
        iqr_ratio = (summary.p75 - summary.p25) / summary.median if summary.median > 0 else float('inf')
        
        return {
            "coefficient_of_variation": cv,
            "iqr_ratio": iqr_ratio,
            "ci_width": summary.ci_upper - summary.ci_lower,
            "n_runs": summary.n,
            "reliability_rating": self._rate_reliability(cv),
        }
    
    def _rate_reliability(self, cv: float) -> str:
        """Rate reliability based on coefficient of variation."""
        if cv < 0.1:
            return "excellent"
        elif cv < 0.2:
            return "good"
        elif cv < 0.3:
            return "acceptable"
        elif cv < 0.5:
            return "poor"
        else:
            return "unreliable"


def generate_leaderboard(
    results_by_config: dict[str, list[EvaluationResult]],
    metric: str = "overall_score",
) -> list[dict[str, Any]]:
    """
    Generate a leaderboard ranking configurations.
    
    Args:
        results_by_config: Mapping of config name to results
        metric: Metric to rank by
        
    Returns:
        Sorted list of configuration performance summaries
    """
    analyzer = StatisticalAnalyzer()
    leaderboard = []
    
    for config, results in results_by_config.items():
        if metric == "overall_score":
            scores = [r.overall_score for r in results]
        else:
            scores = [
                r.get_score(metric.replace("_", " ").title())
                for r in results
            ]
            scores = [s for s in scores if s is not None]
        
        summary = analyzer.summarize(scores)
        leaderboard.append({
            "configuration": config,
            "mean_score": summary.mean,
            "std": summary.std,
            "ci_95": f"[{summary.ci_lower:.2f}, {summary.ci_upper:.2f}]",
            "n_runs": summary.n,
        })
    
    leaderboard.sort(key=lambda x: x["mean_score"], reverse=True)
    
    for rank, entry in enumerate(leaderboard, 1):
        entry["rank"] = rank
    
    return leaderboard

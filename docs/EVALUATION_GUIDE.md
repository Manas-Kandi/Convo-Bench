# Evaluation Guide

This guide explains how ConvoBench evaluates multi-agent workflows and how to interpret results.

## Evaluation Philosophy

ConvoBench evaluation is designed around three key principles:

1. **External Assessment**: The evaluator sits outside the agent chain, ensuring unbiased judgment
2. **Multi-dimensional Scoring**: Performance is measured across multiple orthogonal dimensions
3. **Statistical Validity**: Results are aggregated across multiple runs with proper statistical analysis

## Evaluation Dimensions

### Intent Preservation (Weight: 1.5)

Measures how well the original intent and goals are maintained throughout the workflow.

| Score | Description |
|-------|-------------|
| 5 | Original intent perfectly preserved; final output fully aligns with initial goals |
| 4 | Intent largely preserved with minor drift; goals substantially met |
| 3 | Core intent maintained but some aspects lost or modified |
| 2 | Significant intent drift; goals only partially addressed |
| 1 | Original intent lost or severely distorted; goals not met |

**What to look for:**
- Does the final output address the original request?
- Are the core objectives still recognizable?
- Has the purpose been maintained through all agents?

### Constraint Adherence (Weight: 1.5)

Measures whether specified constraints and requirements are maintained throughout.

| Score | Description |
|-------|-------------|
| 5 | All constraints strictly followed; no violations |
| 4 | Constraints mostly followed; minor technical violations only |
| 3 | Most constraints followed; some non-critical violations |
| 2 | Multiple constraint violations; some critical |
| 1 | Widespread constraint violations; critical requirements ignored |

**What to look for:**
- Are explicit constraints mentioned in the output?
- Were any constraints violated during execution?
- Did agents acknowledge and respect limitations?

### Action Correctness (Weight: 1.0)

Measures whether tool calls and actions are appropriate and correctly executed.

| Score | Description |
|-------|-------------|
| 5 | All actions correct, well-sequenced, and efficient |
| 4 | Actions mostly correct; minor inefficiencies |
| 3 | Actions generally correct; some errors handled |
| 2 | Multiple incorrect actions; poor error handling |
| 1 | Actions largely incorrect or missing; workflow broken |

**What to look for:**
- Were the right tools called for the task?
- Were tool arguments correct?
- Was the sequence of actions logical?

### Coordination Quality (Weight: 1.0)

Measures how well agents coordinate, share information, and work together.

| Score | Description |
|-------|-------------|
| 5 | Seamless coordination; excellent information sharing |
| 4 | Good coordination; information shared effectively |
| 3 | Adequate coordination; some information gaps |
| 2 | Poor coordination; significant information loss |
| 1 | No effective coordination; agents working at cross-purposes |

**What to look for:**
- Did agents build on each other's work?
- Was information passed completely between agents?
- Did agents avoid redundant work?

### Error Propagation (Weight: 1.0)

Measures how errors are handled, contained, and communicated through the chain.

| Score | Description |
|-------|-------------|
| 5 | Errors caught early, contained, and clearly communicated |
| 4 | Errors handled well; minor propagation issues |
| 3 | Errors eventually handled; some unnecessary propagation |
| 2 | Errors poorly handled; significant cascading effects |
| 1 | Errors ignored or amplified; catastrophic propagation |

**What to look for:**
- Were errors detected when they occurred?
- Did errors cascade through subsequent agents?
- Was recovery attempted appropriately?

### Information Fidelity (Weight: 1.5)

Measures accuracy and completeness of information as it passes through agents.

| Score | Description |
|-------|-------------|
| 5 | All information preserved exactly; no loss or distortion |
| 4 | Information largely accurate; minor omissions |
| 3 | Core information preserved; some details lost |
| 2 | Significant information loss or distortion |
| 1 | Critical information lost; severe distortion |

**What to look for:**
- Are numerical values preserved exactly?
- Are names, dates, and identifiers correct?
- Is the structure of information maintained?

## The External Evaluator

### How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Workflow Trace │────▶│    Evaluator    │────▶│   Evaluation    │
│                 │     │      LLM        │     │    Results      │
│  Ground Truth   │────▶│                 │     │                 │
│                 │     │    Rubric       │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Input**: Complete workflow trace + ground truth + rubric
2. **Process**: LLM analyzes trace against expectations
3. **Output**: Scores, evidence, issues, recommendations

### Evaluator Prompt Structure

The evaluator receives:
- Full trace of all workflow steps
- Input/output for each agent
- Tool calls and results
- Ground truth (expected outcomes)
- Scoring rubric with criteria

### Configuring the Evaluator

```python
from convobench.evaluation import ExternalEvaluator, EvaluatorConfig

evaluator = ExternalEvaluator(
    config=EvaluatorConfig(
        model="gpt-4",           # Model for evaluation
        provider="openai",       # Provider
        temperature=0.1,         # Low temp for consistency
        max_tokens=4096,         # Response length
    )
)
```

## Statistical Analysis

### Summary Statistics

For each metric, ConvoBench computes:

| Statistic | Description |
|-----------|-------------|
| Mean | Average score |
| Std | Standard deviation |
| Median | Middle value |
| Min/Max | Range |
| P25/P75 | Interquartile range |
| P95 | 95th percentile |
| CI 95% | 95% confidence interval |

### Comparing Configurations

When comparing two configurations, ConvoBench uses:

1. **Welch's t-test**: Tests if means are significantly different
2. **Cohen's d**: Measures effect size
3. **Significance level**: Default α = 0.05

```python
from convobench.evaluation.analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(significance_level=0.05)
comparison = analyzer.compare_two(scores_a, scores_b, "config_a", "config_b", "overall")

print(f"Mean difference: {comparison.mean_diff:.2f}")
print(f"Effect size (Cohen's d): {comparison.effect_size:.2f}")
print(f"P-value: {comparison.p_value:.4f}")
print(f"Significant: {comparison.significant}")
```

### Reliability Metrics

To assess benchmark reliability:

```python
reliability = analyzer.compute_reliability(results, min_runs=10)

print(f"Coefficient of variation: {reliability['coefficient_of_variation']:.2f}")
print(f"Reliability rating: {reliability['reliability_rating']}")
```

| CV Range | Rating |
|----------|--------|
| < 0.1 | Excellent |
| 0.1 - 0.2 | Good |
| 0.2 - 0.3 | Acceptable |
| 0.3 - 0.5 | Poor |
| > 0.5 | Unreliable |

## Interpreting Results

### Reading a Comparison Report

```json
{
  "configurations": ["gpt-4", "claude-3"],
  "rankings": {
    "overall_score": {"gpt-4": 1, "claude-3": 2},
    "intent_preservation": {"claude-3": 1, "gpt-4": 2}
  },
  "pairwise_comparisons": [
    {
      "config_a": "gpt-4",
      "config_b": "claude-3",
      "metric": "overall_score",
      "mean_difference": 0.3,
      "effect_size": 0.45,
      "p_value": 0.023,
      "statistically_significant": true,
      "winner": "gpt-4"
    }
  ]
}
```

**Interpretation:**
- GPT-4 ranks #1 overall, Claude-3 ranks #2
- Claude-3 is better at intent preservation
- The overall difference is statistically significant (p < 0.05)
- Effect size of 0.45 indicates a "small to medium" practical difference

### Effect Size Guidelines

| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

### Sample Size Recommendations

| Purpose | Minimum Runs |
|---------|--------------|
| Quick test | 5 |
| Standard benchmark | 30 |
| Publication-quality | 100+ |

## Custom Rubrics

### Creating a Custom Rubric

```python
from convobench.evaluation.rubrics import EvaluationRubric, RubricDimension, ScoreLevel

custom_rubric = EvaluationRubric(
    name="Domain-Specific Evaluation",
    description="Evaluation for specialized use case",
    dimensions=[
        RubricDimension(
            name="Domain Accuracy",
            description="Correctness of domain-specific content",
            weight=2.0,
            score_descriptions={
                ScoreLevel.EXCELLENT: "All domain facts correct",
                ScoreLevel.GOOD: "Minor domain inaccuracies",
                ScoreLevel.ACCEPTABLE: "Some domain errors",
                ScoreLevel.POOR: "Significant domain errors",
                ScoreLevel.FAILING: "Fundamentally incorrect",
            }
        ),
        # Add more dimensions...
    ]
)
```

### Using Custom Rubrics

```python
result = await evaluator.evaluate(
    trace=workflow_trace,
    ground_truth=ground_truth,
    rubric=custom_rubric,  # Use custom rubric
)
```

## Best Practices

### 1. Run Sufficient Trials

```python
# Bad: Too few runs
result = await bench.run(scenario, agents, runs=3)

# Good: Statistically meaningful
result = await bench.run(scenario, agents, runs=30)
```

### 2. Use Appropriate Baselines

Compare against:
- Random baseline (lower bound)
- Human performance (upper bound)
- Previous best model

### 3. Report Confidence Intervals

```python
summary = analyzer.summarize(scores)
print(f"Score: {summary.mean:.2f} ± {summary.ci_upper - summary.ci_lower:.2f}")
```

### 4. Check for Ceiling/Floor Effects

If all scores are 5/5 or 1/5, the scenario may be too easy or too hard.

### 5. Validate Evaluator Consistency

Run the same trace through the evaluator multiple times to check consistency.

## Troubleshooting

### Low Inter-run Consistency

**Symptoms**: High coefficient of variation, wide confidence intervals

**Solutions**:
- Increase number of runs
- Reduce scenario randomness
- Check for non-deterministic agent behavior

### Evaluator Disagreement

**Symptoms**: Scores don't match human judgment

**Solutions**:
- Review and refine rubric descriptions
- Provide more specific ground truth
- Use a more capable evaluator model

### Ceiling Effects

**Symptoms**: All configurations score near maximum

**Solutions**:
- Increase scenario difficulty
- Add more challenging constraints
- Use longer agent chains

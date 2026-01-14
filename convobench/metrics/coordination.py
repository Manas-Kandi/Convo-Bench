"""Coordination metrics for multi-agent workflows.

Deterministic, trace-derived metrics meant to map to Agent Factors constructs.

Implemented:
- disagreement_rate: how often successive agents differ materially
- redundant_work_rate: how often successive outputs are near-identical
- stale_state_usage: heuristic based on 'stale', 'outdated', timestamps
- handoff_completeness: overlap between what is asked vs passed along

Note: These are lightweight heuristics. They are designed to be stable and
paper-friendly; scenario-specific metrics can refine them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from convobench.core.types import WorkflowTrace


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> set[str]:
    return {t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text).split() if t}


@dataclass
class CoordinationMetrics:
    disagreement_rate: float
    redundant_work_rate: float
    stale_state_usage_rate: float
    avg_handoff_completeness: float


def compute_coordination_metrics(trace: WorkflowTrace) -> CoordinationMetrics:
    if not trace.steps:
        return CoordinationMetrics(0.0, 0.0, 0.0, 0.0)

    disagreements = 0
    redundancies = 0
    stale_mentions = 0
    handoff_scores = []

    prev_out = None

    for step in trace.steps:
        out = step.output_message.content if step.output_message else ""
        inp = step.input_message.content if step.input_message else ""

        if "stale" in out.lower() or "outdated" in out.lower() or "timestamp" in out.lower():
            stale_mentions += 1

        # handoff completeness: overlap of input tokens retained in output
        in_tok = _tokenize(inp)
        out_tok = _tokenize(out)
        if in_tok:
            handoff_scores.append(_jaccard(in_tok, out_tok))

        if prev_out is not None:
            prev_tok = _tokenize(prev_out)
            sim = _jaccard(prev_tok, out_tok)
            # if very similar, redundant
            if sim > 0.9:
                redundancies += 1
            # if very dissimilar, disagreement/coordination breakdown
            if sim < 0.3:
                disagreements += 1

        prev_out = out

    n_pairs = max(1, len(trace.steps) - 1)

    return CoordinationMetrics(
        disagreement_rate=disagreements / n_pairs,
        redundant_work_rate=redundancies / n_pairs,
        stale_state_usage_rate=stale_mentions / max(1, len(trace.steps)),
        avg_handoff_completeness=sum(handoff_scores) / len(handoff_scores) if handoff_scores else 0.0,
    )

"""Standardized reporting for ConvoBench (AF Benchmark Report).

Exports:
- JSON report schema (AFBenchmarkReport)
- Markdown exporter

This is designed for paper-ready reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from convobench.bench import BenchmarkResult


class AFBenchmarkReport(BaseModel):
    report_version: str = "0.1"
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    benchmark_id: str
    code_version: Optional[str] = None
    scenario_pack_version: Optional[str] = None

    summary: Dict[str, Any]
    aggregate_metrics: Optional[Dict[str, Any]] = None
    comparison_report: Optional[Dict[str, Any]] = None

    traces_path: Optional[str] = None
    evaluations_path: Optional[str] = None


def build_report(result: BenchmarkResult) -> AFBenchmarkReport:
    summary = result.to_dict()
    return AFBenchmarkReport(
        benchmark_id=result.benchmark_id,
        code_version=(result.run_manifest.code_version if result.run_manifest else None),
        scenario_pack_version=(result.run_manifest.scenario_pack_version if result.run_manifest else None),
        summary=summary,
        aggregate_metrics=result.aggregate_metrics.to_dict() if result.aggregate_metrics else None,
        comparison_report=result.comparison_report.to_dict() if result.comparison_report else None,
    )


def export_report_json(result: BenchmarkResult, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(result)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def export_report_markdown(result: BenchmarkResult, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(result)

    lines: List[str] = []
    lines.append(f"# ConvoBench AF Benchmark Report")
    lines.append("")
    lines.append(f"- Benchmark ID: `{report.benchmark_id}`")
    if report.code_version:
        lines.append(f"- Code version: `{report.code_version}`")
    if report.scenario_pack_version:
        lines.append(f"- Scenario pack: `{report.scenario_pack_version}`")
    lines.append(f"- Generated at: `{report.generated_at}`")
    lines.append("")

    if report.aggregate_metrics:
        lines.append("## Aggregate Metrics")
        lines.append("```json")
        lines.append(json.dumps(report.aggregate_metrics, indent=2, default=str))
        lines.append("```")
        lines.append("")

    if report.comparison_report:
        lines.append("## Comparison Report")
        lines.append("```json")
        lines.append(json.dumps(report.comparison_report, indent=2, default=str))
        lines.append("```")
        lines.append("")

    lines.append("## Summary")
    lines.append("```json")
    lines.append(json.dumps(report.summary, indent=2, default=str))
    lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)

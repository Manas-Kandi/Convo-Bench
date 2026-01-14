import json
from pathlib import Path

from convobench.bench import BenchmarkConfig, ConvoBench, create_mock_agents
from convobench.leaderboard import validate_submission
from convobench.reporting import export_report_json, export_report_markdown
from convobench.scenarios.relay import InformationRelay
from convobench.spec import ScenarioPack, ScenarioRef


def test_reporting_exports(tmp_path: Path):
    scenario = InformationRelay(chain_length=2, message_complexity="simple")
    scenario.set_seed(1)

    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=1, save_traces=False, verbose=False))
    agents = create_mock_agents(2)
    result = __import__("asyncio").run(bench.run(scenario, agents, runs=1))

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    export_report_json(result, json_path)
    export_report_markdown(result, md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["benchmark_id"] == result.benchmark_id
    assert "AF Benchmark Report" in md_path.read_text(encoding="utf-8")


def test_leaderboard_submission_validation():
    pack = ScenarioPack(
        pack_id="p",
        name="pack",
        version="0.1.0",
        scenarios=[ScenarioRef(scenario_type="information_relay", params={"chain_length": 2})],
    )

    sub = {
        "team": "team",
        "scenario_pack_id": pack.pack_id,
        "scenario_pack_version": pack.version,
        "model_ids": ["mock"],
        "metrics": {"overall": 3.0},
    }

    parsed = validate_submission(sub)
    assert parsed.team == "team"

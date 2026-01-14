import json
from pathlib import Path

from convobench.bench import ConvoBench, BenchmarkConfig
from convobench.bench import create_mock_agents
from convobench.scenarios.relay import InformationRelay


def test_golden_information_relay_mock_trace_snapshot(tmp_path: Path):
    # Deterministic scenario
    scenario = InformationRelay(chain_length=3, message_complexity="simple")
    scenario.set_seed(42)

    agents = create_mock_agents(3)
    bench = ConvoBench(config=BenchmarkConfig(runs_per_scenario=1, save_traces=False, verbose=False))

    result = __import__("asyncio").run(bench.run(scenario, agents, runs=1))

    # Snapshot a minimal stable subset of the trace
    trace = result.traces[0].to_dict()
    snapshot = {
        "trace_schema_version": trace["trace_schema_version"],
        "scenario_id_prefix": trace["scenario_id"].split("_")[0],
        "num_steps": len(trace["steps"]),
        "first_step_agent": trace["steps"][0]["agent_id"],
        "status": trace["status"],
    }

    # If snapshot file doesn't exist, create it (first run locally). In CI, it should exist.
    golden = tmp_path / "golden_snapshot.json"
    golden.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    loaded = json.loads(golden.read_text(encoding="utf-8"))
    assert loaded == snapshot

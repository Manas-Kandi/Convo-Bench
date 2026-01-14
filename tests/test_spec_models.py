import pytest

from convobench.spec import (
    AFVariables,
    EnvironmentVariables,
    RunManifest,
    ScenarioPack,
    ScenarioRef,
    ToolConfig,
)


def test_afvariables_defaults_validate():
    af = AFVariables()
    assert af.schema_version == "0.1"
    assert af.environment.tool_failure_rate == 0.0


def test_environment_variables_validation():
    EnvironmentVariables(tool_failure_rate=0.5, tool_latency_ms=10)
    with pytest.raises(Exception):
        EnvironmentVariables(tool_failure_rate=1.5)


def test_scenario_pack_requires_scenarios():
    with pytest.raises(ValueError):
        ScenarioPack(pack_id="p1", name="Empty", scenarios=[])


def test_scenario_pack_roundtrip_json():
    pack = ScenarioPack(
        pack_id="relay_pack",
        name="Relay Pack",
        version="0.1.0",
        scenarios=[ScenarioRef(scenario_type="information_relay", params={"chain_length": 3})],
    )
    data = pack.model_dump()
    pack2 = ScenarioPack.model_validate(data)
    assert pack2.pack_id == pack.pack_id


def test_run_manifest_validates_temperature_range():
    RunManifest(
        run_id="r1",
        temperatures=[0.0, 1.0, 2.0],
        scenarios=[ScenarioRef(scenario_type="information_relay", params={})],
    )
    with pytest.raises(ValueError):
        RunManifest(run_id="r2", temperatures=[-0.1])


def test_run_manifest_validates_seeds_non_negative():
    RunManifest(run_id="r1", seeds={"scenario": 123})
    with pytest.raises(ValueError):
        RunManifest(run_id="r2", seeds={"scenario": -1})


def test_tool_config_validation():
    ToolConfig(failure_rate=0.0, latency_ms=0)
    with pytest.raises(Exception):
        ToolConfig(failure_rate=2.0)

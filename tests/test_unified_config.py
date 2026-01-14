import pytest

from convobench.config import BenchmarkRunConfig, ScenarioRunConfig
from convobench.spec import ScenarioPack, ScenarioRef


def test_config_requires_scenario_or_pack():
    cfg = BenchmarkRunConfig()
    with pytest.raises(ValueError):
        cfg.resolved_scenarios()


def test_config_resolves_single_scenario():
    cfg = BenchmarkRunConfig(scenario=ScenarioRunConfig(scenario_type="information_relay", params={"chain_length": 2}))
    scenarios = cfg.resolved_scenarios()
    assert scenarios[0].scenario_type == "information_relay"


def test_config_resolves_pack():
    pack = ScenarioPack(
        pack_id="p",
        name="pack",
        scenarios=[ScenarioRef(scenario_type="information_relay", params={"chain_length": 2})],
    )
    cfg = BenchmarkRunConfig(scenario_pack=pack)
    scenarios = cfg.resolved_scenarios()
    assert scenarios[0].scenario_type == "information_relay"

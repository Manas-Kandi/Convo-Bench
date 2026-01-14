import pytest

from convobench.scenario_pack import instantiate_scenarios
from convobench.spec import ScenarioPack, ScenarioRef


def test_instantiate_scenarios_success():
    pack = ScenarioPack(
        pack_id="p1",
        name="Pack",
        scenarios=[ScenarioRef(scenario_type="information_relay", params={"chain_length": 2})],
    )
    scenarios = instantiate_scenarios(pack)
    assert len(scenarios) == 1
    assert scenarios[0].config.chain_length == 2


def test_instantiate_scenarios_unknown_raises():
    pack = ScenarioPack(
        pack_id="p1",
        name="Pack",
        scenarios=[ScenarioRef(scenario_type="does_not_exist", params={})],
    )
    with pytest.raises(ValueError):
        instantiate_scenarios(pack)

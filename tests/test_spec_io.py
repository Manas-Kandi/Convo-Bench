from pathlib import Path

from convobench.spec import ScenarioPack, ScenarioRef
from convobench.utils.spec_io import load_json, save_json


def test_save_load_json_roundtrip(tmp_path: Path):
    pack = ScenarioPack(
        pack_id="p1",
        name="Pack",
        scenarios=[ScenarioRef(scenario_type="information_relay", params={"chain_length": 2})],
    )
    path = tmp_path / "pack.json"
    save_json(pack, path)
    loaded = load_json(ScenarioPack, path)
    assert loaded.pack_id == "p1"
    assert loaded.scenarios[0].scenario_type == "information_relay"

from pathlib import Path

from convobench.store import RunStore


def test_run_store_roundtrip(tmp_path: Path):
    db = tmp_path / "cb.db"
    store = RunStore(str(db))

    run_id = "run_test"
    store.upsert_run(run_id, "now", "pending", {"a": 1}, {"m": 2})

    got = store.get_run(run_id)
    assert got is not None
    assert got.run_id == run_id
    assert got.status == "pending"
    assert got.config_json["a"] == 1

    store.add_trace(run_id, "wf1", "scenario", {"trace": True})
    traces = store.list_traces(run_id)
    assert traces[0]["trace"] is True

    store.add_metrics(run_id, "wf1", "scenario", {"m": 1})
    metrics = store.list_metrics(run_id)
    assert metrics[0]["m"] == 1

    store.add_evaluation(run_id, "wf1", "scenario", "mock", {"score": 3})
    evals = store.list_evaluations(run_id)
    assert evals[0]["score"] == 3

    store.add_artifact(run_id, "report", "path/to/report.md", {"k": "v"})

    store.close()

from uuid import uuid4

from convobench.core.types import WorkflowTrace


def test_trace_has_schema_version_in_dict():
    trace = WorkflowTrace(workflow_id=uuid4(), scenario_id="s")
    d = trace.to_dict()
    assert d["trace_schema_version"] == "0.1"

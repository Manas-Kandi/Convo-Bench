import pytest

from convobench.scenarios.realism import (
    FailureModeSuite,
    InterruptDrivenPlanning,
    PartialObservabilityReconciliation,
    ProtocolHandoffExperiment,
)


def test_partial_observability_deterministic_seed():
    s1 = PartialObservabilityReconciliation(chain_length=3)
    s1.set_seed(123)
    i1 = s1.create_instance()

    s2 = PartialObservabilityReconciliation(chain_length=3)
    s2.set_seed(123)
    i2 = s2.create_instance()

    assert i1.ground_truth == i2.ground_truth
    assert i1.environment.state.get("doc_count") == 3


def test_interrupt_driven_planning_interrupt_delivered_once():
    s = InterruptDrivenPlanning(num_agents=3)
    s.set_seed(1)
    inst = s.create_instance()

    env = inst.environment
    tool = env.get_tool("get_interrupt")
    assert tool is not None

    first = tool.handler({}, env.state)
    second = tool.handler({}, env.state)

    assert first.get("type") == "new_requirement"
    assert second.get("status") == "none"


def test_protocol_handoff_checksum_stable():
    s = ProtocolHandoffExperiment(chain_length=3, protocol="checksum_readback")
    s.set_seed(7)
    inst = s.create_instance()
    gt = inst.ground_truth

    assert gt["expected_final_output"]["protocol"] == "checksum_readback"
    assert "checksum" in gt["expected_final_output"]


def test_failure_mode_suite_deterministic_seed():
    s1 = FailureModeSuite(chain_length=4, mode="spec_drift")
    s1.set_seed(42)
    i1 = s1.create_instance()

    s2 = FailureModeSuite(chain_length=4, mode="spec_drift")
    s2.set_seed(42)
    i2 = s2.create_instance()

    assert i1.ground_truth == i2.ground_truth

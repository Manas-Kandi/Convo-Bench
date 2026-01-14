from uuid import uuid4

from convobench.core.types import (
    Action,
    ActionType,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    WorkflowStep,
    WorkflowTrace,
    WorkflowStatus,
)
from convobench.metrics.coordination import compute_coordination_metrics
from convobench.metrics.tooling import compute_tool_reliability


def test_tool_reliability_basic_success_and_retry():
    t = WorkflowTrace(workflow_id=uuid4(), scenario_id="s", status=WorkflowStatus.COMPLETED)
    t.metadata["available_tools"] = ["search", "fetch"]

    # Step 0: tool call fails
    call = ToolCall(tool_name="search", arguments={"q": "x"})
    t.steps.append(
        WorkflowStep(
            step_number=0,
            agent_id="a0",
            input_message=Message(role=MessageRole.USER, content="hi"),
            output_message=Message(role=MessageRole.ASSISTANT, content="ok"),
            actions=[Action(action_type=ActionType.TOOL_CALL, agent_id="a0", tool_call=call)],
            tool_results=[ToolResult(tool_call_id=call.id, success=False, result=None, error="timeout")],
            duration_ms=10,
        )
    )

    # Step 1: same tool call succeeds (retry)
    call2 = ToolCall(tool_name="search", arguments={"q": "x"})
    t.steps.append(
        WorkflowStep(
            step_number=1,
            agent_id="a0",
            input_message=Message(role=MessageRole.USER, content="retry"),
            output_message=Message(role=MessageRole.ASSISTANT, content="ok"),
            actions=[Action(action_type=ActionType.TOOL_CALL, agent_id="a0", tool_call=call2)],
            tool_results=[ToolResult(tool_call_id=call2.id, success=True, result={"ok": True})],
            duration_ms=15,
        )
    )

    m = compute_tool_reliability(t)
    assert m.total_tool_calls == 2
    assert m.successful_tool_calls == 1
    assert m.retry_attempts >= 1
    assert m.first_failure_to_next_success_ms is not None
    assert m.incorrect_tool_calls == 0


def test_tool_reliability_incorrect_and_unsafe():
    t = WorkflowTrace(workflow_id=uuid4(), scenario_id="s", status=WorkflowStatus.COMPLETED)
    t.metadata["available_tools"] = ["search"]

    call = ToolCall(tool_name="drop_database", arguments={})
    t.steps.append(
        WorkflowStep(
            step_number=0,
            agent_id="a0",
            input_message=Message(role=MessageRole.USER, content="hi"),
            output_message=Message(role=MessageRole.ASSISTANT, content="ok"),
            actions=[Action(action_type=ActionType.TOOL_CALL, agent_id="a0", tool_call=call)],
            tool_results=[],
            duration_ms=1,
        )
    )

    m = compute_tool_reliability(t)
    assert m.incorrect_tool_calls == 1
    assert m.unsafe_tool_attempts == 1


def test_coordination_metrics_disagreement_and_redundancy():
    t = WorkflowTrace(workflow_id=uuid4(), scenario_id="s", status=WorkflowStatus.COMPLETED)

    t.steps.append(
        WorkflowStep(
            step_number=0,
            agent_id="a0",
            input_message=Message(role=MessageRole.USER, content="Do X"),
            output_message=Message(role=MessageRole.ASSISTANT, content="We should do X then Y"),
            duration_ms=1,
        )
    )
    t.steps.append(
        WorkflowStep(
            step_number=1,
            agent_id="a1",
            input_message=Message(role=MessageRole.USER, content="Do X"),
            output_message=Message(role=MessageRole.ASSISTANT, content="We should do X then Y"),
            duration_ms=1,
        )
    )
    t.steps.append(
        WorkflowStep(
            step_number=2,
            agent_id="a2",
            input_message=Message(role=MessageRole.USER, content="Do X"),
            output_message=Message(role=MessageRole.ASSISTANT, content="Ignore X. Do Z."),
            duration_ms=1,
        )
    )

    m = compute_coordination_metrics(t)
    assert m.redundant_work_rate > 0
    assert m.disagreement_rate > 0
    assert 0.0 <= m.avg_handoff_completeness <= 1.0
